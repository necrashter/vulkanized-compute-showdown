#ifndef MODEL_H
#define MODEL_H

#include "VulkanContext.h"
#include "Texture.h"

#include <vulkan/vulkan.hpp>

#define GLM_FORCE_RADIANS
// use 0, 1 depth in Vulkan instead of OpenGL's -1 to 1
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// Required for make_vec4, make_mat4, etc.
#include <glm/gtc/type_ptr.hpp>

#include "tiny_gltf.h"

#include <iostream>

class VertexAttr {
public:
    std::string name;
    vk::Format format;
    size_t size;

    const uint8_t* buffer;

    VertexAttr(const std::string& name, vk::Format format);

    size_t loadBuffer(const tinygltf::Primitive& primitive, const tinygltf::Model& input);

    inline void fill(void* target, size_t index) {
        if (buffer) {
            memcpy(target, (void*)&buffer[index*size], size);
        }
    }
};


class Model {
public:
    VulkanContext* const context;

    std::vector<VertexAttr> vertexAttributes;
    size_t totalOffset = 0;
    std::vector<size_t> vertexOffsets;

    inline void addVertexAttribute(const std::string& name, vk::Format format) {
        vertexOffsets.push_back(totalOffset);
        vertexAttributes.emplace_back(name, format);
        totalOffset += vertexAttributes.back().size;
    }

    inline vk::VertexInputBindingDescription getVertexInputBindingDescription(uint32_t binding=0) {
        return vk::VertexInputBindingDescription(
                binding, // binding
                totalOffset, // stride
                vk::VertexInputRate::eVertex);
    }

    inline std::vector<vk::VertexInputAttributeDescription> getVertexAttributeDescriptions(uint32_t binding=0) {
        std::vector<vk::VertexInputAttributeDescription> out;
        for (uint32_t i = 0; i < vertexAttributes.size(); ++i) {
            out.emplace_back(
                    i, binding, // location and binding
                    vertexAttributes[i].format,
                    vertexOffsets[i]
                    );
        }
        return out;
    }

    struct {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
    } vertices;

    struct {
        uint32_t count;
        vk::Buffer buffer;
        vk::DeviceMemory memory;
    } indices;


    /*
       Scene Structure
   */

    // A primitive contains the data for a single draw call
    struct Primitive {
        uint32_t firstIndex;
        uint32_t indexCount;
        int32_t materialIndex;
    };

    // A node represents an object in the gltf scene graph
    struct Node;
    struct Node {
        std::string name;
        Node* parent;
        glm::mat4 matrix;
        std::vector<Node> children;
        std::vector<Primitive> primitives;
    };

    // A gltf material stores information in e.g. the texture that is attached to it and colors
    struct Material {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        uint32_t baseColorTextureIndex;
    };

    // A gltf texture stores a reference to the image and a sampler
    // In this sample, we are only interested in the image
    struct Texture {
        int32_t imageIndex;
        Texture(int32_t imageIndex): imageIndex(imageIndex) {}
    };

    /*
       Model data
       */
    std::vector<TextureImage> images;
    std::vector<Texture> textures;
    std::vector<Material> materials;
    std::vector<Node> nodes;

    std::vector<vk::DescriptorSet> materialDescriptorSets;

    // These are used while reading the model.
    // They are cleaned after this data is uploaded to GPU
    uint32_t totalVertexCount = 0;
    std::vector<uint8_t> vertexData;
    std::vector<uint32_t> indexData;


    Model(VulkanContext* context): context(context) {
    }

private:
    void loadImages(tinygltf::Model& input);
    void loadTextures(tinygltf::Model& input);
    void loadMaterials(tinygltf::Model& input);
    void loadNode(
            const tinygltf::Node& inputNode,
            const tinygltf::Model& input,
            Node* parent);

public:
    void loadFile(const char* filename);

    // After loading the model, create vertex and index buffers
    void createBuffers();

    vk::DescriptorSetLayout createMaterialDescriptorSetLayout();
    void createMaterialDescriptorSets(
            vk::DescriptorPool descriptorPool,
            vk::DescriptorSetLayout layout);

    // Draw a single node including child nodes (if present)
    void renderNode(vk::CommandBuffer commandBuffer, vk::PipelineLayout pipelineLayout,
        Node& node, glm::mat4 matrix);

    // Draw the gltf scene starting at the top-level-nodes
    void render(vk::CommandBuffer commandBuffer,
            vk::PipelineLayout pipelineLayout, glm::mat4 matrix = glm::mat4(1.0f));

    Node* getNode(const std::string& name);

    /*
       CLENAUP
       */

    void cleanup() {
        context->device->destroyBuffer(vertices.buffer);
        context->device->freeMemory(vertices.memory);
        context->device->destroyBuffer(indices.buffer);
        context->device->freeMemory(indices.memory);
        for (auto image : images) {
            image.cleanup(context);
        }
    }
};

#endif
