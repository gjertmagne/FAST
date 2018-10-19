#include "FAST/Algorithms/CoherentPointDrift/CoherentPointDrift.hpp"
#include "FAST/SceneGraph.hpp"
#include "CoherentPointDrift.hpp"

#undef min
#undef max
#include <limits>
#include <random>
#include <unordered_set>

//#include <Eigen/SVD>
#include <iostream>
#include <ctime>

namespace fast {

    CoherentPointDrift::CoherentPointDrift() {
        createInputPort<Mesh>(0);
        createInputPort<Mesh>(1);
        createOutputPort<Mesh>(0);
        mVariance = 100;
        mScale = 1.0;
        mW = 0.0;
        mIteration = 0;
        mMaxIterations = 100;
        mTolerance = 1e-4;
        mIterationError = mTolerance + 1.0;
        mRandomSamplingPoints = 0;
        mDistanceThreshold = -1;
        timeE = 0.0;
        timeM = 0.0;
        mTransformationType = CoherentPointDrift::RIGID;
        mTransformation = AffineTransformation::New();
    }


    void CoherentPointDrift::setFixedMeshPort(DataPort::pointer port) {
        setInputConnection(0, port);
    }

    void CoherentPointDrift::setMovingMeshPort(DataPort::pointer port) {
        setInputConnection(1, port);
    }

    void CoherentPointDrift::setFixedMesh(Mesh::pointer data) {
        setInputData(0, data);
    }

    void CoherentPointDrift::setMovingMesh(Mesh::pointer data) {
        setInputData(1, data);
    }

    AffineTransformation::pointer CoherentPointDrift::getOutputTransformation() {
        return mTransformation;
    }

    void CoherentPointDrift::execute() {
        auto fixedMesh = getInputData<Mesh>(0);
        auto movingMesh = getInputData<Mesh>(1);

        // Get access to the two point sets
        MeshAccess::pointer accessFixedSet = fixedMesh->getMeshAccess(ACCESS_READ);
        MeshAccess::pointer accessMovingSet = movingMesh->getMeshAccess(ACCESS_READ);

        // Get the points from the meshes
        std::vector<MeshVertex> fixedVertices = accessFixedSet->getVertices();
        std::vector<MeshVertex> movingVertices = accessMovingSet->getVertices();

        // Set dimensions of point sets
        unsigned int numDimensionsFixed = (unsigned int)fixedVertices[0].getPosition().size();
        unsigned int numDimensionsMoving = (unsigned int)movingVertices[0].getPosition().size();
        assert(numDimensionsFixed == numDimensionsMoving);
        mNumDimensions = numDimensionsFixed;
        mNumFixedPoints = (unsigned int)fixedVertices.size();
        mNumMovingPoints = (unsigned int)movingVertices.size();

        // Store point sets in matrices
        MatrixXf fixedPoints = MatrixXf::Zero(mNumFixedPoints, mNumDimensions);
        MatrixXf movingPoints = MatrixXf::Zero(mNumMovingPoints, mNumDimensions);
        for(int i = 0; i < mNumFixedPoints; ++i) {
            fixedPoints.row(i) = fixedVertices[i].getPosition();
        }
        for(int i = 0; i < mNumMovingPoints; ++i) {
            movingPoints.row(i) = movingVertices[i].getPosition();
        }

        // Homogeneous coordinates?
//        MatrixXf fixedPointsHomog = fixedPoints.rowwise().homogeneous();
//        MatrixXf movingPointsHomog = movingPoints.rowwise().homogeneous();

        // Store original moving point set
        MatrixXf movingPointsOriginal = movingPoints;

        // Apply existing transformation (for testing) to moving point cloud
        auto existingTransform = SceneGraph::getEigenAffineTransformationFromData(movingMesh);
        std::cout << "Existing transform: \n" << existingTransform.affine() << std::endl;
        movingPoints = movingPoints * existingTransform.linear().transpose();
        movingPoints += existingTransform.translation().transpose().replicate(mNumMovingPoints, 1);

        // Testing
        std::cout << "\n****************************************\n";
        std::cout << "mNumFixedPoints = " << mNumFixedPoints
                  << ", mNumMovingPoints = " << mNumMovingPoints << std::endl;
        std::cout << "Dimension = " << mNumDimensions << std::endl;

        // Initialize the variance
        if (mVariance == 100) {
            mVariance = (   (double)mNumMovingPoints * (fixedPoints.transpose() * fixedPoints).trace() +
                            (double)mNumFixedPoints * (movingPoints.transpose() * movingPoints).trace() -
                            2.0 * fixedPoints.colwise().sum() * movingPoints.colwise().sum().transpose()  ) /
                    (double)(mNumFixedPoints * mNumMovingPoints * mNumDimensions);
            double varianceTemp = mVariance;
            std::cout << "Variance, initial: " << varianceTemp << std::endl;
        }

        // Calculate the value of the objective function
        mObjectiveFunction = -mIterationError - double(mNumFixedPoints * mNumDimensions)/2 * log(mVariance);

        /* *************
         * Normalization
         * ************/
        // Center point clouds around origin, zero mean
        MatrixXf movingMean = movingPoints.colwise().sum() / mNumMovingPoints;
        MatrixXf fixedMean = fixedPoints.colwise().sum() / mNumFixedPoints;
        fixedPoints -= fixedMean.replicate(mNumFixedPoints, 1);
        movingPoints -= movingMean.replicate(mNumMovingPoints, 1);


        // Scale point clouds to have unit variance
        double fixedScale = sqrt(fixedPoints.cwiseProduct(fixedPoints).sum() / (double)mNumFixedPoints);
        double movingScale = sqrt(movingPoints.cwiseProduct(movingPoints).sum() / (double)mNumMovingPoints);
        std::cout << "Fixed scale factor: " << fixedScale << std::endl;
        std::cout << "Moving scale factor: " << movingScale << std::endl;
        fixedPoints /= fixedScale;
        movingPoints /= movingScale;


        /* *************************
         * Get some points drifting!
         * ************************/
        // Initialize the probability matrix of correspondences
        mProbabilityMatrix = MatrixXf::Zero(mNumMovingPoints, mNumFixedPoints);

        clock_t startEM = clock();
        double timeStartEM = omp_get_wtime();
        while (mIteration < mMaxIterations && mIterationError > mTolerance) {
            expectation(&mProbabilityMatrix, &fixedPoints, &movingPoints);
            maximization(&mProbabilityMatrix, &fixedPoints, &movingPoints);
            mIteration++;
        }
        clock_t endEM = clock();
        double timeEndEM = omp_get_wtime();
//        double timeEM = (double) (endEM-startEM) / CLOCKS_PER_SEC;
        double totalTimeEM = timeEndEM - timeStartEM;

        int itTemp = mIteration;
        std::cout << "EM converged in " << itTemp-1 << " iterations in " << totalTimeEM << " s.\n";
        std::cout << "Time spent on expectation: " << timeE << " s\n";
        std::cout << "Time spent on maximization: " << timeM/1000.0 << " s" << std::endl;
        std::cout << "Remaining time spent on updating transform and point clouds\n";


        /* ***********
         * Denormalize
         * **********/
        mScale *= fixedScale / movingScale;

        auto transform = AffineTransformation::New();
        Affine3f registrationTransform = Affine3f::Identity();

        mTransformation->getTransform().scale(float(mScale));
        Vector3f initialTranslation = fixedMean.transpose() - mTransformation->getTransform().linear() * movingMean.transpose();
        registrationTransform = mTransformation->getTransform();

        Affine3f denormalizationTranslation = Affine3f::Identity();
        denormalizationTranslation.translate(initialTranslation);

        Affine3f registrationTransformTotal = denormalizationTranslation * registrationTransform;
        transform->setTransform(registrationTransformTotal*existingTransform);
        movingMesh->getSceneGraphNode()->setTransformation(transform);
        addOutputData(0, movingMesh);

        std::cout << "\n*****************************************\n";
        std::cout << "Final registration matrix: \n" << registrationTransformTotal.matrix() << std::endl;
        std::cout << "Registered transform * existingTransform (should be identity): \n"
            << registrationTransformTotal * existingTransform.matrix() << std::endl;

    }




    void CoherentPointDrift::expectation(
                    MatrixXf* probabilityMatrix,
                    MatrixXf* fixedPoints, MatrixXf* movingPoints) {

        clock_t startE = clock();
        double timeStartE = omp_get_wtime();

        // Calculate distances between the points in the two point sets
//        MatrixXf dist = MatrixXf::Zero(mNumMovingPoints, mNumFixedPoints);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < mNumMovingPoints; ++i) {
            for (int j = 0; j < mNumFixedPoints; ++ j) {
                VectorXf fixedVec = fixedPoints->row(j);
                VectorXf movingVec = movingPoints->row(i);
                VectorXf diff = fixedVec - movingVec;
                probabilityMatrix->row(i)[j] = diff.squaredNorm();
                // Let row i in P equal the squared distances from all fixed points to moving point i
            }
        }



        // Working fine
//        for (int i = 0; i < mNumMovingPoints; ++i) {
//            MatrixXf movingPointMatrix = movingPoints->row(i).replicate(mNumFixedPoints, 1);
//            dist = *fixedPoints - movingPointMatrix;                 // Distance between all fixed points and moving point i
//            dist = dist.cwiseAbs2();                                // Square distance components (3xN)
//            probabilityMatrix->row(i) = dist.rowwise().sum();       // Sum x, y, z components (1xN)
//                 Let row i in P equal the squared distances from all fixed points to moving point i
//        }


        // Testing parallelization with OpenMP
//        int i;
//        MatrixXf movingPointMatrix;
//        #pragma omp parallel num_threads(8)
//        {
//            #pragma omp for private(i, movingPointMatrix, dist)
//            for (i = 0; i < mNumMovingPoints; ++i) {
//                movingPointMatrix = movingPoints->row(i).replicate(mNumFixedPoints, 1);
//                dist = *fixedPoints -
//                       movingPointMatrix;                // Distance between all fixed points and moving point i
//                dist = dist.cwiseAbs2();                                // Square distance components (3xN)
//                probabilityMatrix->row(i) = dist.rowwise().sum();       // Sum x, y, z components (1xN)
//                 Let row i in P equal the squared distances from all fixed points to moving point i
//            }
//        }





        // Normal distribution
        double c = pow(2*(double)EIGEN_PI*mVariance, (double)mNumDimensions/2)
                   * (mW/(1-mW)) * ((double)mNumMovingPoints/(double)mNumFixedPoints);

        *probabilityMatrix *= -1.0/(2.0 * mVariance);
        *probabilityMatrix = probabilityMatrix->array().exp();

        MatrixXf denominatorRow = probabilityMatrix->colwise().sum();
        denominatorRow =  denominatorRow.array() + c;

        // Ensure that one does not divide by zero
        MatrixXf shouldBeLargerThanEpsilon = Eigen::NumTraits<float>::epsilon() * MatrixXf::Ones(1, mNumFixedPoints);
        denominatorRow = denominatorRow.cwiseMax(shouldBeLargerThanEpsilon);
        MatrixXf denominator = denominatorRow.replicate(mNumMovingPoints, 1);

        *probabilityMatrix = probabilityMatrix->cwiseQuotient(denominator);

        clock_t endE = clock();
        double timeEndE = omp_get_wtime();
        timeE += timeEndE - timeStartE;
//        timeE += (double) (endE-startE) / CLOCKS_PER_SEC * 1000.0;
    }

    void CoherentPointDrift::maximization(MatrixXf* probabilityMatrix,
            MatrixXf* fixedPoints, MatrixXf* movingPoints) {

        clock_t startM = clock();

        // Define some useful matrix sums
        mPt1 = probabilityMatrix->transpose().rowwise().sum();      // mNumFixedPoints x 1
        mP1 = probabilityMatrix->rowwise().sum();                   // mNumMovingPoints x 1
        mNp = probabilityMatrix->sum();                             // 1

        MatrixXf muX = fixedPoints->transpose() * mPt1 / mNp;
        MatrixXf muY = movingPoints->transpose() * mP1 / mNp;

        MatrixXf fixedPointsPred = *fixedPoints - muX.transpose().replicate(mNumFixedPoints, 1);
        MatrixXf movingPointsPred = *movingPoints - muY.transpose().replicate(mNumMovingPoints, 1);

        // Single value decomposition (SVD)
        const MatrixXf A = fixedPointsPred.transpose() * probabilityMatrix->transpose() * movingPointsPred;
        auto svdU =  A.bdcSvd(Eigen::ComputeThinU);
        auto svdV =  A.bdcSvd(Eigen::ComputeThinV);
        auto S = svdU.singularValues();
        const MatrixXf U = svdU.matrixU();
        const MatrixXf V = svdV.matrixV();

        MatrixXf UVt = U * V.transpose();
        Eigen::RowVectorXf C = Eigen::RowVectorXf::Ones(mNumDimensions);
        C[mNumDimensions-1] = UVt.determinant();


        /* ************************************************************
         * Find transformation parameters: rotation, scale, translation
         * ***********************************************************/
        mRotation = U * C.asDiagonal() * V.transpose();
        MatrixXf AtR = A.transpose() * mRotation;
        MatrixXf ARt = A * mRotation.transpose();
        double traceAtR = AtR.trace();
        double traceXPX = (fixedPointsPred.transpose() * mPt1.asDiagonal() * fixedPointsPred).trace();
        double traceYPY = (movingPointsPred.transpose() * mP1.asDiagonal() * movingPointsPred).trace();

        mScale = traceAtR / traceYPY;
        mTranslation = muX - mScale * mRotation * muY;

        // Update variance
        mVariance = ( traceXPX - mScale * traceAtR ) / (mNp * mNumDimensions);
        if (mVariance <= 0) {
            mVariance = mTolerance / 10;
        }

        clock_t endM = clock();
        timeM += (double) (endM-startM) / CLOCKS_PER_SEC * 1000.0;

        /* ****************
         * Update transform
         * ***************/
        Affine3f iterationTransform = Affine3f::Identity();
        iterationTransform.translation() = Vector3f(mTranslation);
        iterationTransform.linear() = mRotation;
        iterationTransform.scale(float(mScale));

        Affine3f currentRegistrationTransform;
        MatrixXf registrationMatrix = iterationTransform.matrix() * mTransformation->getTransform().matrix();
        currentRegistrationTransform.matrix() = registrationMatrix;
        mTransformation->setTransform(currentRegistrationTransform);


        /* *************************
         * Transform the point cloud
         * ************************/
        MatrixXf movingPointsTransformed =
                mScale * *movingPoints * mRotation.transpose() + mTranslation.transpose().replicate(mNumMovingPoints, 1);
        *movingPoints = movingPointsTransformed;


        /* ******************************************
         * Calculate change in the objection function
         * *****************************************/
        double objectiveFunctionOld = mObjectiveFunction;
        mObjectiveFunction =
                (traceXPX - 2 * mScale * ARt.trace() + mScale * mScale * traceYPY) / (2 * mVariance)
                + (mNp * mNumDimensions)/2 * log(mVariance);
        mIterationError = abs(mObjectiveFunction - objectiveFunctionOld);

//        std::cout << "Change in error this iteration: " << mIterationError << std::endl;
//        std::cout << "Total transformation matrix so far:\n"
//                  << mTransformation->getTransform().matrix() << std::endl;
    }


    void CoherentPointDrift::setTransformationType(const CoherentPointDrift::TransformationType type) {
        mTransformationType = type;
    }
}