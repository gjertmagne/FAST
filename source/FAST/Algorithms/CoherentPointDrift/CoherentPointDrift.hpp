#ifndef COHERENT_POINT_DRIFT_HPP
#define COHERENT_POINT_DRIFT_HPP

#include "FAST/AffineTransformation.hpp"
#include "FAST/ProcessObject.hpp"
#include "FAST/Data/Mesh.hpp"

namespace fast {

    class FAST_EXPORT  CoherentPointDrift: public ProcessObject {
    FAST_OBJECT(CoherentPointDrift)
    public:
        typedef enum { RIGID, TRANSLATION } TransformationType;
        void setFixedMeshPort(DataPort::pointer port);
        void setFixedMesh(Mesh::pointer data);
        void setMovingMeshPort(DataPort::pointer port);
        void setMovingMesh(Mesh::pointer data);
        void setTransformationType(const CoherentPointDrift::TransformationType type);
        AffineTransformation::pointer getOutputTransformation();
    private:
        CoherentPointDrift();
        void expectation(MatrixXf* probabilityMatrix,
                         MatrixXf* fixedPoints, MatrixXf* movingPoints);
        void maximization(MatrixXf* probabilityMatrix,
                          MatrixXf* fixedPoints, MatrixXf* movingPoints);
        void transformPointCloud();
        void execute();

        MatrixXf mProbabilityMatrix;            // P
        MatrixXf mPt1;                          // Colwise sum of P, then transpose
        MatrixXf mP1;                           // Rowwise sum of P
        MatrixXf mRotation;                     // R
        MatrixXf mTranslation;                  // t
        unsigned int mNumFixedPoints;           // N
        unsigned int mNumMovingPoints;          // M
        unsigned int mNumDimensions;            // D
        double mObjectiveFunction;              // Q
        double mScale;                          // s
        double mVariance;                       // sigma^2
        double mIterationError;                 // Change in error from iteration to iteration
        double mTolerance;                      // Convergence criteria for EM iterations
        double mNp;                             // Sum of all elements in P
        float mW;                               // Weight of the uniform distribution
        unsigned char mIteration;
        unsigned char mMaxIterations;
        double timeE;
        double timeM;
        int mRandomSamplingPoints;
        float mDistanceThreshold;
        AffineTransformation::pointer mTransformation;
        CoherentPointDrift::TransformationType mTransformationType;
    };

} // end namespace fast

#endif
