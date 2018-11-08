#ifndef COHERENT_POINT_DRIFT_HPP
#define COHERENT_POINT_DRIFT_HPP

#include "FAST/AffineTransformation.hpp"
#include "FAST/ProcessObject.hpp"
#include "FAST/Data/Mesh.hpp"

#include "FAST/SmartPointers.hpp"

namespace fast {

    class FAST_EXPORT  CoherentPointDrift: public ProcessObject {
//    FAST_OBJECT(CoherentPointDrift)
    public:
        typedef enum { RIGID, AFFINE, NONRIGID } TransformationType;
        void setFixedMeshPort(DataPort::pointer port);
        void setFixedMesh(Mesh::pointer data);
        void setMovingMeshPort(DataPort::pointer port);
        void setMovingMesh(Mesh::pointer data);
        void setMaximumIterations(unsigned char maxIterations);
        void setUniformWeight(float uniformWeight);
        void setTolerance(double tolerance);
        AffineTransformation::pointer getOutputTransformation();

    protected:
        CoherentPointDrift();
        void execute();
        MatrixXf mFixedPoints;
        MatrixXf mMovingPoints;
        std::shared_ptr<Mesh> mFixedMesh;
        std::shared_ptr<Mesh> mMovingMesh;
        MatrixXf mMovingMeanInitial;
        MatrixXf mFixedMeanInitial;
        unsigned int mNumFixedPoints;           // N
        unsigned int mNumMovingPoints;          // M
        unsigned int mNumDimensions;            // D
        double mFixedNormalizationScale;
        double mMovingNormalizationScale;
        double mTolerance;                      // Convergence criteria for EM iterations
        float mUniformWeight;                   // Weight of the uniform distribution
        double mScale;                          // s
        AffineTransformation::pointer mTransformation;
        bool mRegistrationConverged;
        double mTimeStart;
        double timeE;
        double timeEDistances;
        double timeENormal;
        double timeEPosterior;
        double timeM;
        double timeMUseful;
        double timeMCenter;
        double timeMSVD;
        double timeMParameters;
        double timeMUpdate;
    private:
        virtual void expectation(MatrixXf& fixedPoints, MatrixXf& movingPoints) = 0;
        virtual void maximization(MatrixXf& fixedPoints, MatrixXf& movingPoints) = 0;
        virtual void initializeVarianceAndMore() = 0;
        void initializePointSets();
        void applyExistingTransform(Affine3f existingTransform);
        void printCloudDimensions();
        void normalizePointSets();
        void denormalizePointSets();

//        MatrixXf mProbabilityMatrix;            // P
//        MatrixXf mPt1;                          // Colwise sum of P, then transpose
//        MatrixXf mP1;                           // Rowwise sum of P
//        MatrixXf mRotation;                     // R
//        MatrixXf mTranslation;                  // t

//        float mNp;                              // Sum of all elements in P
        unsigned char mIteration;
        unsigned char mMaxIterations;
        CoherentPointDrift::TransformationType mTransformationType;
    };

} // end namespace fast

#endif
