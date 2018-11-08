#include <FAST/Visualization/SimpleWindow.hpp>
#include <FAST/Importers/VTKMeshFileImporter.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>
#include "FAST/Testing.hpp"
#include "FAST/Visualization/VertexRenderer/VertexRenderer.hpp"
#include "CoherentPointDrift.hpp"
#include "Rigid.hpp"

#include <random>
#include <iostream>
using namespace fast;

Mesh::pointer getPointCloud() {
    auto importer = VTKMeshFileImporter::New();
    importer->setFilename(Config::getTestDataPath() + "Surface_LV.vtk");
    auto port = importer->getOutputPort();
    importer->update(0);
    return port->getNextFrame<Mesh>();
}

void modifyPointCloud(Mesh::pointer &pointCloud, double fractionOfPointsToKeep, double noiseSampleRatio=0.0) {
    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();

    // Sample the preferred amount of points from the point cloud
    auto numVertices = (unsigned int) vertices.size();
    auto numSamplePoints = (unsigned int) ceil(fractionOfPointsToKeep * numVertices);
    std::vector<MeshVertex> newVertices;

    std::unordered_set<int> movingIndices;
    unsigned int sampledPoints = 0;
    std::default_random_engine distributionEngine;
    std::uniform_int_distribution<unsigned int> distribution(0, numVertices-1);
    while (sampledPoints < numSamplePoints) {
        unsigned int index = distribution(distributionEngine);
        if (movingIndices.count(index) < 1) {
            newVertices.push_back(vertices.at(index));
            movingIndices.insert(index);
            ++sampledPoints;
        }
    }

    // Add noise to point cloud
    auto numNoisePoints = (unsigned int) ceil(noiseSampleRatio * numSamplePoints);
    float minX, minY, minZ;
    Vector3f position0 = vertices[0].getPosition();
    minX = position0[0];
    minY = position0[1];
    minZ = position0[2];
    float maxX = minX, maxY = minY, maxZ = minZ;
    for (auto &vertex : vertices) {
        Vector3f position = vertex.getPosition();
        if (position[0] < minX) {minX = position[0]; }
        if (position[0] > maxX) {maxX = position[0]; }
        if (position[1] < minY) {minY = position[1]; }
        if (position[1] > maxY) {maxY = position[1]; }
        if (position[2] < minZ) {minZ = position[2]; }
        if (position[2] > maxZ) {maxZ = position[2]; }
    }

    std::uniform_real_distribution<float> distributionNoiseX(minX, maxX);
    std::uniform_real_distribution<float> distributionNoiseY(minY, maxY);
    std::uniform_real_distribution<float> distributionNoiseZ(minZ, maxZ);

    for (int noiseAdded = 0; noiseAdded < numNoisePoints; noiseAdded++) {
        float noiseX = distributionNoiseX (distributionEngine);
        float noiseY = distributionNoiseY (distributionEngine);
        float noiseZ = distributionNoiseZ (distributionEngine);
        Vector3f noisePosition = Vector3f(noiseX, noiseY, noiseZ);
        MeshVertex noise = MeshVertex(noisePosition, Vector3f(1, 0, 0), Color::Black());
        newVertices.push_back(noise);
    }

    // Update point cloud to include the removed points and added noise
    pointCloud->create(newVertices);
}


TEST_CASE("cpd", "[fast][coherentpointdrift][visual][cpd]") {

    // Load identical point clouds
    auto cloud1 = getPointCloud();
    auto cloud2 = getPointCloud();
    auto cloud3 = getPointCloud();

    // Modify point clouds
    float fractionOfPointsToKeep = 0.8;
    float noiseLevel = 0.5;
    modifyPointCloud(cloud2, fractionOfPointsToKeep, noiseLevel);
    modifyPointCloud(cloud3, fractionOfPointsToKeep, noiseLevel);

    // Set registration settings
    float uniformWeight = 0.5;
    double tolerance = 1e-4;

    // Create transformation for moving point cloud
    Vector3f translation(-0.04f, 0.05f, -0.02f);
    auto transform = AffineTransformation::New();
    Affine3f affine = Affine3f::Identity();
//    affine.translate(translation);
    affine.rotate(Eigen::AngleAxisf(3.141592f / 3.0f, Eigen::Vector3f::UnitY()));
    affine.scale(0.5);
    transform->setTransform(affine);

    // Apply transform to one point cloud
    cloud2->getSceneGraphNode()->setTransformation(transform);

    // Apply transform to a point cloud not registered (for reference)
    cloud3->getSceneGraphNode()->setTransformation(transform);

    // Run for different numbers of iterations
    std::vector<unsigned char> iterations = {100};
    for(auto maxIterations : iterations) {

        // Run Coherent Point Drift
        std::cout << "Ready to generate new rigid\n";
        auto cpd = CoherentPointDriftRigid::New();
        std::cout << "Rigid object created\n";
        cpd->setFixedMesh(cloud1);
        cpd->setMovingMesh(cloud2);
        cpd->setMaximumIterations(maxIterations);
        cpd->setTolerance(tolerance);
        cpd->setUniformWeight(uniformWeight);

        auto renderer = VertexRenderer::New();
        renderer->addInputData(cloud1, Color::Green(), 3.0);                        // Fixed points
        renderer->addInputData(cloud3, Color::Blue(), 2.0);                         // Moving points
        renderer->addInputConnection(cpd->getOutputPort(), Color::Red(), 2.0);      // Moving points registered

        auto window = SimpleWindow::New();
        window->addRenderer(renderer);
        //window->setTimeout(1000);
        window->start();
    }

}
