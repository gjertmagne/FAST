#include <FAST/Visualization/SimpleWindow.hpp>
#include <FAST/Importers/VTKMeshFileImporter.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>
#include "FAST/Testing.hpp"
#include "FAST/Visualization/VertexRenderer/VertexRenderer.hpp"
#include "CoherentPointDrift.hpp"

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

void modifyPointCloud(Mesh::pointer pointCloud, double fractionOfPointsToKeep, double fractionNoisePoints) {
    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();

    auto numVertices = (unsigned int) vertices.size();
    auto numSamplePoints = (unsigned int) ceil(fractionOfPointsToKeep * numVertices);
//    std::vector<MeshVertex> newVertices[numSamplePoints];
    std::vector<MeshVertex> newVertices;

    std::unordered_set<int> movingIndices;
    unsigned int sampledPoints = 0;
    std::default_random_engine distributionEngine;
    std::uniform_int_distribution<unsigned int> distribution(0, numVertices-1);
    while (sampledPoints < numSamplePoints) {
        unsigned int index = distribution(distributionEngine);
        if (movingIndices.count(index) < 1) {
//            newVertices[sampledPoints].insert(vertices[index]);
//            newVertices->at(sampledPoints) = vertices.at(index);
            newVertices.push_back(vertices.at(index));
            movingIndices.insert(index);
            ++sampledPoints;
        }
    }
    pointCloud->create(newVertices);
}


TEST_CASE("cpd", "[fast][coherentpointdrift][visual][cpd]") {

    // Load identical point clouds
    auto cloud1 = getPointCloud();
    auto cloud2 = getPointCloud();
    auto cloud3 = getPointCloud();

    // Modify point clouds
    double fractionOfPointsToKeep = 0.5;
    double noiseLevel = 0.0;
    modifyPointCloud(cloud2, fractionOfPointsToKeep, noiseLevel);
    modifyPointCloud(cloud3, fractionOfPointsToKeep, noiseLevel);

    std::vector<unsigned char> iterations = {0, 50};
    for(auto maxIterations : iterations) {
        // Create transformation for moving point cloud
        Vector3f translation(-0.04f, 0.05f, -0.02f);
        auto transform = AffineTransformation::New();
        Affine3f affine = Affine3f::Identity();
        affine.translate(translation);
        affine.rotate(Eigen::AngleAxisf(3.14f / 3.0f, Eigen::Vector3f::UnitY()));
        affine.scale(0.6);
        transform->setTransform(affine);


        // Apply transform to one point cloud
        cloud2->getSceneGraphNode()->setTransformation(transform);

        // Apply transform to a point cloud not registered
        cloud3->getSceneGraphNode()->setTransformation(transform);

        // Run Coherent Point Drift
        auto cpd = CoherentPointDrift::New();
        cpd->setFixedMesh(cloud1);
        cpd->setMovingMesh(cloud2);
        cpd->setMaximumIterations(maxIterations);

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
