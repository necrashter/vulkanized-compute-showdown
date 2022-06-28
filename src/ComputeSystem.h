#pragma once

#include "VulkanContext.h"
#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>


class ComputeSystem {
    struct ComputeStorage {
        size_t size;
        vk::Buffer buffer;
        vk::DeviceMemory memory;

        ComputeStorage(): size(0), buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE) {}
    };

    struct ComputeUniform {
        size_t size;
        void* mapping;
        vk::Buffer buffer;
        vk::DeviceMemory memory;

        ComputeUniform(): size(0), mapping(nullptr), buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE) {}
    };

private:
    VulkanContext* const context;

    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;
    vk::PipelineLayout pipelineLayout;

    // Need special command buffer and hence command pool
    vk::CommandPool commandPool;
    vk::CommandBuffer commandBuffer;

    // Shader storage buffer objects
    std::vector<ComputeStorage> storageBuffers;
    // Uniform buffers
    std::vector<ComputeUniform> uniformBuffers;

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    std::vector<vk::DescriptorType> descriptorTypes;

    std::vector<vk::Pipeline> pipelines;

public:
    vk::Semaphore sem;


    // Barriers to acquire the shader storage buffer for graphics pipeline
    std::vector<vk::BufferMemoryBarrier> graphicsAcquireBarriers;
    // Barriers to release the shader storage buffer for graphics pipeline
    std::vector<vk::BufferMemoryBarrier> graphicsReleaseBarriers;

    // Initializes the compute system with given Vulkan context.
    ComputeSystem(VulkanContext* context);

    // Create SSBO (Shader Storage Object) initialized with the given data.
    // Return pointer to create vk::Buffer (for binding as vertex buffer)
    vk::Buffer* createShaderStorage(const void* data, size_t size);

    // Create uniform storage buffer
    // Return pointer (binding point for uniform buffer)
    void* createUniformBuffer(size_t size);

    // After creating buffers, call this to finalize the plpeline layout.
    // Creates descriptor sets, etc.
    void finalizeLayout();

    // Add a pipeline (compute shader pass) constructed from given shader code and entry point.
    void addPipeline(const std::vector<char>& shaderCode, const char* entryPoint);

    // Record dispatch command after the pipeline is ready.
    void recordCommands(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

    // Signal the semaphore for kickstart
    void signalSemaphore();

    inline vk::CommandBuffer* const getCommandBufferPointer() {
        return &commandBuffer;
    }

    ~ComputeSystem();
};

