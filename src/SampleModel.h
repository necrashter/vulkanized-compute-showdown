#ifndef SAMPLE_MODEL_H
#define SAMPLE_MODEL_H

#include "VulkanContext.h"

#include <vulkan/vulkan.hpp>

#define GLM_FORCE_RADIANS
// use 0, 1 depth in Vulkan instead of OpenGL's -1 to 1
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 uv;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return std::array<vk::VertexInputAttributeDescription, 3> {
            vk::VertexInputAttributeDescription(
                    0, 0,
                    vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),

            vk::VertexInputAttributeDescription(
                    1, 0, // location and binding
                    vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),

            vk::VertexInputAttributeDescription(
                    2, 0, // location and binding
                    vk::Format::eR32G32Sfloat, offsetof(Vertex, uv)),
        };
    }
};


typedef uint16_t index_t;
const vk::IndexType vkindex_t = vk::IndexType::eUint16;


class SampleModel {
private:
    const std::vector<Vertex> vertexData = {
        {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
        {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
        {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

        {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
        {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
        {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
    };

    const std::vector<index_t> indexData = {
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
    };


    struct {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
    } vertices;

    struct {
        uint32_t count;
        vk::Buffer buffer;
        vk::DeviceMemory memory;
    } indices;


public:
    VulkanContext* context;

    SampleModel(VulkanContext* context): context(context) {
    }

    inline vk::VertexInputBindingDescription getBindingDescription() {
        return Vertex::getBindingDescription();
    }

    inline std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return Vertex::getAttributeDescriptions();
    }

	/**
	 * Creates vertex and index buffers.
	 */
    void createBuffers() {
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
    }

    void render(vk::CommandBuffer commandBuffer) {
        vk::Buffer vertexBuffers[] = { vertices.buffer };
        vk::DeviceSize offsets[] = { 0 };
        commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
        commandBuffer.bindIndexBuffer(indices.buffer, 0, vkindex_t);
        commandBuffer.drawIndexed(indices.count, 1, 0, 0, 0);
    }

    void cleanup() {
        context->device->destroyBuffer(vertices.buffer);
        context->device->freeMemory(vertices.memory);
        context->device->destroyBuffer(indices.buffer);
        context->device->freeMemory(indices.memory);
    }
};

#endif
