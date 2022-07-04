#include "Model.h"
#include "log.h"

/**
 * VERTEX ATTRIBUTES
 */

size_t getFormatSize(vk::Format format) {
    switch(format) {
        case vk::Format::eR32Sfloat:
        case vk::Format::eR32Sint:
        case vk::Format::eR32Uint:
            return 4;
        case vk::Format::eR32G32Sfloat:
        case vk::Format::eR32G32Sint:
        case vk::Format::eR32G32Uint:
            return 8;
        case vk::Format::eR32G32B32Sfloat:
        case vk::Format::eR32G32B32Sint:
        case vk::Format::eR32G32B32Uint:
            return 12;
        case vk::Format::eR32G32B32A32Sfloat:
        case vk::Format::eR32G32B32A32Sint:
        case vk::Format::eR32G32B32A32Uint:
            return 16;
        default:
            throw std::runtime_error("Unknown format in getFormatSize");
    }
}

VertexAttr::VertexAttr(const std::string& name, vk::Format format):
    name(name), format(format), size(getFormatSize(format)), buffer(nullptr) {
}

size_t VertexAttr::loadBuffer(const tinygltf::Primitive& primitive, const tinygltf::Model& input) {
    auto it = primitive.attributes.find(name);
    if (it != primitive.attributes.end()) {
        const tinygltf::Accessor& accessor = input.accessors[it->second];
        const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
        buffer = (const uint8_t*)&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
        return accessor.count;
    } else {
        buffer = nullptr;
        return 0;
    }
}


/*
   LOADING
   */

void Model::loadImages(tinygltf::Model& input) {
    for (size_t i = 0; i < input.images.size(); i++) {
        TextureImage loadedImage;
        tinygltf::Image& gltfImage = input.images[i];
        // Get the image data from the gltf loader
        unsigned char* buffer = nullptr;
        VkDeviceSize bufferSize = 0;
        bool deleteBuffer = false;
        // We convert RGB-only images to RGBA; most devices don't support RGB-formats in Vulkan
        if (gltfImage.component == 3) {
            bufferSize = gltfImage.width * gltfImage.height * 4;
            buffer = new unsigned char[bufferSize];
            unsigned char* rgba = buffer;
            unsigned char* rgb = &gltfImage.image[0];
            size_t maxi = gltfImage.width * gltfImage.height;
            for (size_t i = 0; i < maxi; ++i) {
                memcpy(rgba, rgb, sizeof(unsigned char) * 3);
                rgba += 4;
                rgb += 3;
            }
            deleteBuffer = true;
        } else {
            buffer = &gltfImage.image[0];
            bufferSize = gltfImage.image.size();
        }
        // Load texture from image buffer
        loadedImage.loadFromBuffer(context, buffer, bufferSize, gltfImage.width, gltfImage.height, vk::Format::eR8G8B8A8Unorm);
        images.push_back(loadedImage);

        if (deleteBuffer) {
            delete[] buffer;
        }
    }
}


void Model::loadTextures(tinygltf::Model& input) {
    for (auto t : input.textures) {
        textures.emplace_back(t.source);
    }
}

void Model::loadMaterials(tinygltf::Model& input) {
    for (auto gltfMat : input.materials) {
        Material mat;
        // Get the base color factor
        if (gltfMat.values.find("baseColorFactor") != gltfMat.values.end()) {
            mat.baseColorFactor = glm::make_vec4(gltfMat.values["baseColorFactor"].ColorFactor().data());
        }
        // Get base color texture index
        if (gltfMat.values.find("baseColorTexture") != gltfMat.values.end()) {
            mat.baseColorTextureIndex = gltfMat.values["baseColorTexture"].TextureIndex();
        }
        materials.push_back(mat);
    }
}

namespace {
    template<typename T>
    inline void addIndices(const void* buffer, std::vector<uint32_t>& indexData, const size_t count, const uint32_t vertexStart) {
        const T* buf = reinterpret_cast<const T*>(buffer);
        for (size_t index = 0; index < count; index++) {
            indexData.push_back(buf[index] + vertexStart);
        }
    }
}

void Model::loadNode(
        const tinygltf::Node& inputNode,
        const tinygltf::Model& input,
        Node* parent) {
    Node node;
    node.name = inputNode.name;
    node.matrix = glm::mat4(1.0f);

    // Get the local node matrix
    // It's either made up from translation, rotation, scale or a 4x4 matrix
    if (inputNode.translation.size() == 3) {
        node.matrix = glm::translate(node.matrix, glm::vec3(glm::make_vec3(inputNode.translation.data())));
    }
    if (inputNode.rotation.size() == 4) {
        glm::quat q = glm::make_quat(inputNode.rotation.data());
        node.matrix *= glm::mat4(q);
    }
    if (inputNode.scale.size() == 3) {
        node.matrix = glm::scale(node.matrix, glm::vec3(glm::make_vec3(inputNode.scale.data())));
    }
    if (inputNode.matrix.size() == 16) {
        node.matrix = glm::make_mat4x4(inputNode.matrix.data());
    };

    // Load node's children
    if (inputNode.children.size() > 0) {
        for (size_t i = 0; i < inputNode.children.size(); i++) {
            loadNode(input.nodes[inputNode.children[i]], input, &node);
        }
    }

    if (inputNode.mesh > -1) {
        const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
        // Iterate through all primitives of this node's mesh
        for (size_t i = 0; i < mesh.primitives.size(); ++i) {
            const tinygltf::Primitive& gltfPrimitive = mesh.primitives[i];
            uint32_t firstIndex = static_cast<uint32_t>(indexData.size());
            uint32_t vertexStart = totalVertexCount;
            uint32_t indexCount = 0;
            // Vertices
            {
                size_t vertexCount = 0;
                for (auto& attr: vertexAttributes) {
                    vertexCount = std::max(vertexCount, attr.loadBuffer(gltfPrimitive, input));
                }
                totalVertexCount += vertexCount;

                // Append data to model's vertex buffer
                for (size_t v = 0; v < vertexCount; ++v) {
                    uint8_t* vert = &(*vertexData.insert(vertexData.end(), totalOffset, 0));
                    for (uint32_t j = 0; j < vertexAttributes.size(); ++j) {
                        vertexAttributes[j].fill(vert + vertexOffsets[j], v);
                    }
                }
            }
            // Indices
            {
                const tinygltf::Accessor& accessor = input.accessors[gltfPrimitive.indices];
                const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];

                indexCount += static_cast<uint32_t>(accessor.count);

                const void* rawBuffer = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                const size_t count = accessor.count;

                switch (accessor.componentType) {
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                        addIndices<uint32_t>(rawBuffer, indexData, count, vertexStart);
                        break;
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                        addIndices<uint16_t>(rawBuffer, indexData, count, vertexStart);
                        break;
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                        addIndices<uint8_t>(rawBuffer, indexData, count, vertexStart);
                        break;
                    default:
                        throw std::runtime_error("Unsupported index component type");
                }
            }
            Primitive primitive{};
            primitive.firstIndex = firstIndex;
            primitive.indexCount = indexCount;
            primitive.materialIndex = gltfPrimitive.material;
            node.primitives.push_back(primitive);
        }
    }

    if (parent) {
        parent->children.push_back(node);
    } else {
        nodes.push_back(node);
    }
}


void Model::loadFile(const char* filename) {
    // TODO: can convert this to ModelManager and allow loading multiple models in it
    if (vertexAttributes.empty()) {
        throw std::runtime_error("No vertex attributes are set in Model");
    }
    tinygltf::Model gltfInput;
    tinygltf::TinyGLTF gltfContext;
    std::string error, warning;

    TLOG("ModelLoader") << "Loading Model: " << filename << std::endl;

    bool fileLoaded = gltfContext.LoadASCIIFromFile(&gltfInput, &error, &warning, filename);

    TLOG("ModelLoader") << "glTF loaded" << std::endl;

    if (fileLoaded) {
        TLOG("ModelLoader") << "Loading images..." << std::endl;
        loadImages(gltfInput);
        TLOG("ModelLoader") << "Loading materials..." << std::endl;
        loadMaterials(gltfInput);
        TLOG("ModelLoader") << "Loading textures..." << std::endl;
        loadTextures(gltfInput);
        TLOG("ModelLoader") << "Loading nodes..." << std::endl;
        const tinygltf::Scene& scene = gltfInput.scenes[0];
        for (size_t i = 0; i < scene.nodes.size(); i++) {
            const tinygltf::Node node = gltfInput.nodes[scene.nodes[i]];
            loadNode(node, gltfInput, nullptr);
        }
    } else {
        std::stringstream s;
        s << "Could not load gltf file: " << filename << '\n';
        s << "Error: " << error << '\n';
        throw std::runtime_error(s.str());
    }

    TLOG("ModelLoader") << "Model loaded to CPU" << std::endl;

    if (!warning.empty()) {
        std::cout << "Warning while reading " << filename << ": " << warning << '\n';
    }
}


////////////////////////////////////////////////////////////////////////
//                           CREATE BUFFERS                           //
////////////////////////////////////////////////////////////////////////

// After loading the model, create vertex and index buffers
void Model::createBuffers() {
    TLOG("ModelLoader") << "Creating vertex and index buffers..." << std::endl;
    vk::DeviceSize vertexBufferSize = sizeof(vertexData[0]) * vertexData.size();
    vk::DeviceSize indexBufferSize = sizeof(indexData[0]) * indexData.size();
    vk::DeviceSize bufferSize = std::max(vertexBufferSize, indexBufferSize);

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    context->createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = context->device->mapMemory(stagingBufferMemory, 0, bufferSize);

    // Vertex Buffer
    context->createBuffer(
            vertexBufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vertices.buffer, vertices.memory);
    memcpy(data, vertexData.data(), (size_t)vertexBufferSize);
    context->copyBuffer(stagingBuffer, vertices.buffer, vertexBufferSize);

    // Index Buffer
    context->createBuffer(
            indexBufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            indices.buffer, indices.memory);
    memcpy(data, indexData.data(), (size_t)indexBufferSize);
    context->copyBuffer(stagingBuffer, indices.buffer, indexBufferSize);

    context->device->unmapMemory(stagingBufferMemory);

    context->device->destroyBuffer(stagingBuffer);
    context->device->freeMemory(stagingBufferMemory);

    indices.count = indexData.size();

    // Don't need these anymore
    vertexData.clear();
    indexData.clear();

    TLOG("ModelLoader") << "Created vertex and index buffers" << std::endl;
}


vk::DescriptorSetLayout Model::createMaterialDescriptorSetLayout() {
    std::array<vk::DescriptorSetLayoutBinding, 1> bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment),
    };
    return context->device->createDescriptorSetLayout({{}, bindings.size(), bindings.data()});
}

void Model::createMaterialDescriptorSets(
        vk::DescriptorPool descriptorPool,
        vk::DescriptorSetLayout layout) {
    try {
        std::vector<vk::DescriptorSetLayout> layouts(materials.size(), layout);
        materialDescriptorSets = context->device->allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo(
                    descriptorPool,
                    materials.size(),
                    layouts.data()));
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (uint32_t i = 0; i < materials.size(); ++i) {
        TextureImage& img = images[textures[materials[i].baseColorTextureIndex].imageIndex];
        vk::DescriptorImageInfo imageInfo(
                img.sampler, img.view,
                vk::ImageLayout::eShaderReadOnlyOptimal);

        std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {
            // Image sampler
            vk::WriteDescriptorSet(
                    materialDescriptorSets[i], 0, 0, 1,
                    vk::DescriptorType::eCombinedImageSampler,
                    &imageInfo,
                    nullptr
                    ),
        };
        context->device->updateDescriptorSets(
                descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }
}


/*
   RENDERING
   */

void Model::renderNode(vk::CommandBuffer commandBuffer, vk::PipelineLayout pipelineLayout,
        Node& node, glm::mat4 matrix) {
    matrix = matrix * node.matrix;
    if (node.primitives.size() > 0) {
        commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4), &matrix);
        for (Primitive& primitive : node.primitives) {
            if (primitive.indexCount > 0) {
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, 1, &materialDescriptorSets[primitive.materialIndex], 0, nullptr);
                commandBuffer.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, 0, 0);
            }
        }
    }
    for (auto& child : node.children) {
        renderNode(commandBuffer, pipelineLayout, child, matrix);
    }
}

void Model::render(vk::CommandBuffer commandBuffer,
        vk::PipelineLayout pipelineLayout, glm::mat4 matrix) {
    // All vertices and indices are stored in single buffers, so we only need to bind once
    VkDeviceSize offsets[1] = { 0 };
    commandBuffer.bindVertexBuffers(0, 1, &vertices.buffer, offsets);
    commandBuffer.bindIndexBuffer(indices.buffer, 0, vk::IndexType::eUint32);
    // Render all nodes at top-level
    for (auto& node : nodes) {
        renderNode(commandBuffer, pipelineLayout, node, matrix);
    }
}


Model::Node* Model::getNode(const std::string& name) {
    for (auto& node : nodes) {
        if (node.name == name) return &node;
    }
    return nullptr;
}

