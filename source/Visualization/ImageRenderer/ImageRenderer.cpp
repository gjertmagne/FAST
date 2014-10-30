#include "ImageRenderer.hpp"
#include "Exception.hpp"
#include "DeviceManager.hpp"
#include "HelperFunctions.hpp"
#include "Image.hpp"
#include "SceneGraph.hpp"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl_gl.h>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#else
#if _WIN32
#include <GL/gl.h>
#include <CL/cl_gl.h>
#else
#include <GL/glx.h>
#include <CL/cl_gl.h>
#endif
#endif

using namespace fast;

#ifndef GL_RGBA32F // this is missing on windows and mac for some reason
#define GL_RGBA32F 0x8814 
#endif

void ImageRenderer::execute() {
    if(!mInput.isValid())
        throw Exception("No input was given to ImageRenderer");

    Image::pointer input;
    if(mInput->isDynamicData()) {
        input = DynamicImage::pointer(mInput)->getNextFrame();
    } else {
        input = mInput;
    }

    if(input->getDimensions() != 2)
        throw Exception("The ImageRenderer only supports 2D images");

    mImageToRender = input;
    // Determine level and window
    float window = mWindow;
    float level = mLevel;
    // If mWindow/mLevel is equal to -1 use default level/window values
    if(window == -1) {
        window = getDefaultIntensityWindow(input->getDataType());
    }
    if(level == -1) {
        level = getDefaultIntensityLevel(input->getDataType());
    }
	
    setOpenGLContext(mDevice->getGLContext());

    OpenCLImageAccess2D access = input->getOpenCLImageAccess2D(ACCESS_READ, mDevice);
    cl::Image2D* clImage = access.get();

    glEnable(GL_TEXTURE_2D);
    if(mTextureIsCreated) {
        // Delete old texture
        glDeleteTextures(1, &mTexture);
    }

	// Resize window to image
    mWidth = input->getWidth();
    mHeight = input->getHeight();

    // Create OpenGL texture
    glGenTextures(1, &mTexture);
    glBindTexture(GL_TEXTURE_2D, mTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, clImage->getImageInfo<CL_IMAGE_WIDTH>(), clImage->getImageInfo<CL_IMAGE_HEIGHT>(), 0, GL_RGBA, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();

    // Create CL-GL image
#if defined(CL_VERSION_1_2)
    mImageGL = cl::ImageGL(
            mDevice->getContext(),
            CL_MEM_READ_WRITE,
            GL_TEXTURE_2D,
            0,
            mTexture
    );
#else
    mImageGL = cl::Image2DGL(
            mDevice->getContext(),
            CL_MEM_READ_WRITE,
            GL_TEXTURE_2D,
            0,
            mTexture
    );
#endif

    // Run kernel to fill the texture
    cl::CommandQueue queue = mDevice->getCommandQueue();
    std::vector<cl::Memory> v;
    v.push_back(mImageGL);
    queue.enqueueAcquireGLObjects(&v);

    recompileOpenCLCode(input);
    mKernel.setArg(0, *clImage);
    mKernel.setArg(1, mImageGL);
    mKernel.setArg(2, level);
    mKernel.setArg(3, window);
    queue.enqueueNDRangeKernel(
            mKernel,
            cl::NullRange,
            cl::NDRange(clImage->getImageInfo<CL_IMAGE_WIDTH>(), clImage->getImageInfo<CL_IMAGE_HEIGHT>()),
            cl::NullRange
    );

    queue.enqueueReleaseGLObjects(&v);
    queue.finish();

    mTextureIsCreated = true;
	releaseOpenGLContext();
}

void ImageRenderer::setInput(ImageData::pointer image) {
    mInput = image;
    setParent(mInput);
    mIsModified = true;
}

void ImageRenderer::recompileOpenCLCode(Image::pointer input) {
    // Check if code has to be recompiled
    bool recompile = false;
    if(!mTextureIsCreated) {
        recompile = true;
    } else {
        if(mTypeCLCodeCompiledFor != input->getDataType())
            recompile = true;
    }
    if(!recompile)
        return;
    std::string buildOptions = "";
    if(input->getDataType() == TYPE_FLOAT) {
        buildOptions = "-DTYPE_FLOAT";
    } else if(input->getDataType() == TYPE_INT8 || input->getDataType() == TYPE_INT16) {
        buildOptions = "-DTYPE_INT";
    } else {
        buildOptions = "-DTYPE_UINT";
    }
    int i = mDevice->createProgramFromSource(std::string(FAST_SOURCE_DIR) + "/Visualization/ImageRenderer/ImageRenderer.cl", buildOptions);
    mKernel = cl::Kernel(mDevice->getProgram(i), "renderToTexture");
    mTypeCLCodeCompiledFor = input->getDataType();
}

ImageRenderer::ImageRenderer() : Renderer() {
    mDevice = DeviceManager::getInstance().getDefaultVisualizationDevice();
    mTextureIsCreated = false;
    mIsModified = true;
    mDoTransformations = true;
    mScale = 1.0f;
    mWidth = 0;
    mHeight = 0;
}

void ImageRenderer::draw() {
    if(!mTextureIsCreated)
        return;

    //setOpenGLContext(mDevice->getGLContext());

    if(mDoTransformations) {
        SceneGraph& graph = SceneGraph::getInstance();
        SceneGraphNode::pointer node = graph.getDataNode(mImageToRender);
        LinearTransformation transform = graph.getLinearTransformationFromNode(node);

        float matrix[16] = {
                transform(0,0), transform(1,0), transform(2,0), transform(3,0),
                transform(0,1), transform(1,1), transform(2,1), transform(3,1),
                transform(0,2), transform(1,2), transform(2,2), transform(3,2),
                transform(0,3), transform(1,3), transform(2,3), transform(3,3)
        };

        glMultMatrixf(matrix);
    }

    glBindTexture(GL_TEXTURE_2D, mTexture);

//glColor3f(1.0,0,0);
    glBegin(GL_QUADS);
        glTexCoord2i(0, 1);
        glVertex3f(0, mHeight, 0.0f);
        glTexCoord2i(1, 1);
        glVertex3f(mWidth, mHeight, 0.0f);
        glTexCoord2i(1, 0);
        glVertex3f(mWidth, 0, 0.0f);
        glTexCoord2i(0, 0);
        glVertex3f(0, 0, 0.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

}

void ImageRenderer::keyPressEvent(QKeyEvent* event) {

}

void ImageRenderer::turnOffTransformations() {
    mDoTransformations = false;
}

BoundingBox ImageRenderer::getBoundingBox() {
    BoundingBox inputBoundingBox = mImageToRender->getBoundingBox();
    if(mDoTransformations) {
        SceneGraph& graph = SceneGraph::getInstance();
        SceneGraphNode::pointer node = graph.getDataNode(mImageToRender);
        LinearTransformation transform = graph.getLinearTransformationFromNode(node);
        BoundingBox transformedBoundingBox = inputBoundingBox.getTransformedBoundingBox(transform);
        return transformedBoundingBox;
    } else {
        return inputBoundingBox;
    }
}
