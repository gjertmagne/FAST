#ifndef FAST_RIGID_H
#define FAST_RIGID_H


#include "FAST/AffineTransformation.hpp"
#include "FAST/ProcessObject.hpp"
#include "FAST/Data/Mesh.hpp"
#include "FAST/Algorithms/CoherentPointDrift/CoherentPointDrift.hpp"

namespace fast {

    class FAST_EXPORT CoherentPointDriftRigid: public CoherentPointDrift {
    FAST_OBJECT(CoherentPointDriftRigid);
    public:
        CoherentPointDriftRigid();
        void expectation(MatrixXf& fixedPoints, MatrixXf& movingPoints) override;
        void maximization(MatrixXf& fixedPoints, MatrixXf& movingPoints) override;
        void initializeVarianceAndMore() override;

    private:
        MatrixXf mProbabilityMatrix;            // P
        VectorXf mPt1;                          // Colwise sum of P, then transpose
        VectorXf mP1;                           // Rowwise sum of P
        MatrixXf mRotation;                     // R
        MatrixXf mTranslation;                  // t
        double mObjectiveFunction;              // Q
        double mVariance;                       // sigma^2
        double mIterationError;                 // Change in error from iteration to iteration
        float mNp;                              // Sum of all elements in P
        TransformationType mTransformationType;
        //        double mScale;                          // s

    };

}


#endif //FAST_RIGID_H
