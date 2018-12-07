#include <FAST/Data/SpatialDataObject.hpp>
#include <fstream>
#include "Renderer.hpp"
#include "Window.hpp"

namespace fast {

Renderer::Renderer() {
}


uint Renderer::addInputData(DataObject::pointer data) {
    uint nr = getNrOfInputConnections();
    if(nr > 0)
        createInputPort<DataObject>(nr);
    setInputData(nr, data);
    return nr;
}

uint Renderer::addInputConnection(DataPort::pointer port) {
    uint nr = getNrOfInputConnections();
    if(nr > 0)
        createInputPort<DataObject>(nr);
    setInputConnection(nr, port);
    return nr;
}

void Renderer::lock() {
    mMutex.lock();
}

void Renderer::unlock() {
    mMutex.unlock();
}

void Renderer::stopPipeline() {
    mStop = true;
    mHasRendered = true;
    mRenderedCV.notify_one();
    ProcessObject::stopPipeline();
}

void Renderer::postDraw() {
    mHasRendered = true;
    mRenderedCV.notify_one();
}

void Renderer::execute() {
    std::unique_lock<std::mutex> lock(mMutex);
    if(mStop) {
        return;
    }

    // Check if current images has not been rendered, if not wait
    while(!mHasRendered) {
        mRenderedCV.wait(lock);
    }
    // This simply gets the input data for each connection and puts it into a data structure
    for(uint inputNr = 0; inputNr < getNrOfInputConnections(); inputNr++) {
        if(hasNewInputData(inputNr)) {
            SpatialDataObject::pointer input = getInputData<SpatialDataObject>(inputNr);

            mHasRendered = false;
            mDataToRender[inputNr] = input;
        }
    }
}

BoundingBox Renderer::getBoundingBox(bool transform) {
    std::vector<Vector3f> coordinates;

    if(mDataToRender.size() == 0)
        throw Exception("Renderer has no input data. Unable to create bounding box and thereby initialize GL scene.");

    for(auto it : mDataToRender) {
        BoundingBox transformedBoundingBox;
        if(transform) {
            transformedBoundingBox = it.second->getTransformedBoundingBox();
        } else {
            transformedBoundingBox = it.second->getBoundingBox();
        }

        MatrixXf corners = transformedBoundingBox.getCorners();
        for(uint j = 0; j < 8; j++) {
            coordinates.push_back((Vector3f)corners.row(j));
        }
    }

    if(coordinates.size() == 0)
        throw Exception("Renderer did not get any data. Unable to create bounding box and thereby initialize GL scene.");

    return BoundingBox(coordinates);
}


void Renderer::createShaderProgram(std::vector<std::string> shaderFilenames, std::string programName) {
    // We need an active GL context to do this, and we also need to initialize the OpenGLFunctions
    Window::getMainGLContext()->makeCurrent();
    initializeOpenGLFunctions();

    uint programID = glCreateProgram();
    mShaderProgramIDs[programName] = programID;

    for(std::string filename : shaderFilenames) {
        attachShader(filename, programName);
    }
}


void Renderer::attachShader(std::string filename, std::string programName) {
    // Make sure shader is created
    if(mShaderProgramIDs.count(programName) == 0)
        createShaderProgram({}, programName);

    // Load GLSL Shader from source file
    std::ifstream fd(filename.c_str());
    if(fd.fail()) {
        throw Exception("Unable to read shader program " + filename);
    }
    auto src = std::string(std::istreambuf_iterator<char>(fd),
                           (std::istreambuf_iterator<char>()));

    // Create shader object
    const char * source = src.c_str();
    uint shaderID;
    // Extract file extension and create the correct shader type
    auto idx = filename.rfind(".");
    auto ext = filename.substr(idx + 1);
    /*if(ext == "comp") { // symbol not present on mac
        shaderID = glCreateShader(GL_COMPUTE_SHADER);
    } else*/
    if(ext == "frag") {
        shaderID = glCreateShader(GL_FRAGMENT_SHADER);
    } else if(ext == "geom") {
      shaderID = glCreateShader(GL_GEOMETRY_SHADER);
    } else if(ext == "tcs") {
       shaderID =  glCreateShader(GL_TESS_CONTROL_SHADER);
    } else if(ext == "tes") {
       shaderID =  glCreateShader(GL_TESS_EVALUATION_SHADER);
    } else if(ext == "vert") {
        shaderID =  glCreateShader(GL_VERTEX_SHADER);
    } else {
        throw Exception("Unknown shader extension. Extension should indicate shader type (.vert, .frag, .geom, .tes, .comp)");
    }
    glShaderSource(shaderID, 1, &source, nullptr);
    glCompileShader(shaderID);

    // Display errors
    int status;
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &status);
    if(!status) {
        int length;
        glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &length);
        std::unique_ptr<char[]> buffer(new char[length]);
        glGetShaderInfoLog(shaderID, length, nullptr, buffer.get());
        throw Exception("Unable to compile shader " + filename + ": " + buffer.get());
    }

    // Attach shader and free allocated memory
    glAttachShader(getShaderProgram(programName), shaderID);
    glDeleteShader(shaderID);
}

void Renderer::activateShader(std::string programName) {
    // Check if linked
    int status;
    uint programID = getShaderProgram(programName) ;
    glGetProgramiv(programID, GL_LINK_STATUS, &status);
    if(!status) {
        // Shader is not linked, link it
        glLinkProgram(programID);
        glGetProgramiv(programID, GL_LINK_STATUS, &status);
        if(!status) {
            // Link failed, throw exception
            int length;
            glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &length);
            std::unique_ptr<char[]> buffer(new char[length]);
            glGetProgramInfoLog(programID, length, nullptr, buffer.get());
            throw Exception("Unable to link shader program: " + std::string(buffer.get()));
        }
    }
    glUseProgram(getShaderProgram(programName));
}

void Renderer::deactivateShader() {
    glUseProgram(0);
}

uint Renderer::getShaderProgram(std::string programName) {
    try {
        return mShaderProgramIDs.at(programName);
    } catch(...) {
        throw Exception("Shader program with name " + programName + " not found");
    }
}

void Renderer::setShaderUniform(std::string name, Matrix4f matrix, std::string shaderProgram) {
    glUniformMatrix4fv(getShaderUniformLocation(name, shaderProgram), 1, GL_FALSE, matrix.data());
}

void Renderer::setShaderUniform(std::string name, Affine3f matrix, std::string shaderProgram) {
    setShaderUniform(name, matrix.matrix(), shaderProgram);
}


void Renderer::setShaderUniform(std::string name, Vector3f vector, std::string shaderProgram) {
    glUniform3f(getShaderUniformLocation(name, shaderProgram), vector.x(), vector.y(), vector.z());
}

void Renderer::setShaderUniform(std::string name, float value, std::string shaderProgram) {
    glUniform1f(getShaderUniformLocation(name, shaderProgram), value);
}

void Renderer::setShaderUniform(std::string name, bool value, std::string shaderProgram) {
    glUniform1i(getShaderUniformLocation(name, shaderProgram), value);
}

void Renderer::setShaderUniform(std::string name, int value, std::string shaderProgram) {
    glUniform1i(getShaderUniformLocation(name, shaderProgram), value);
}

int Renderer::getShaderUniformLocation(std::string name, std::string shaderProgram) {
    int location = glGetUniformLocation(getShaderProgram(shaderProgram), name.c_str());
    if(location == -1)
        throw Exception("Unable to find location of matrix4f uniform " + name + " in shader program " + shaderProgram);
    return location;
}

void Renderer::reset() {
    mStop = false;
    mHasRendered = false;
}

}
