#pragma once

#include "VulkanBaseApp.h"
#include <vulkan/vulkan.hpp>
#include "FrameUniform.h"

#include <glm/glm.hpp>


class ComputeSystem {
    class ComputeStorage : public Descriptor {
    public:
        size_t size;
        vk::Buffer buffer;
        vk::DeviceMemory memory;

        ComputeStorage(): Descriptor(eSSBO),
            size(0), buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE) {}
    };

private:
    VulkanBaseApp* const app;

    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;
    vk::PipelineLayout pipelineLayout;

    // Need special command buffer and hence command pool
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;

    // Shader storage buffer objects
    std::vector<ComputeStorage> storageBuffers;
    // Uniform buffers
    std::vector<FrameUniform> uniformBuffers;

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    std::vector<Descriptor*> descriptors;

    std::vector<vk::Pipeline> pipelines;

    std::vector<vk::BufferMemoryBarrier> getMemoryBarriers(vk::BufferMemoryBarrier temp, uint32_t frame);

public:
    vk::Semaphore sem;

    vk::Queue queue;
    uint32_t queueIndex;


    // Barriers to acquire the shader storage buffer for graphics pipeline
    std::vector<vk::BufferMemoryBarrier> graphicsAcquireBarriers[MAX_FRAMES_IN_FLIGHT];
    // Barriers to release the shader storage buffer for graphics pipeline
    std::vector<vk::BufferMemoryBarrier> graphicsReleaseBarriers[MAX_FRAMES_IN_FLIGHT];

    // Initializes the compute system with given Vulkan app.
    ComputeSystem(VulkanBaseApp* app);

    // Create SSBO (Shader Storage Object) initialized with the given data.
    // Return pointer to create vk::Buffer (for binding as vertex buffer)
    vk::Buffer* createShaderStorage(const void* data, size_t size);

    // Create uniform storage buffer
    // Return pointer (binding point for uniform buffer)
    FrameUniform* createUniformBuffer(size_t size);

    // After creating buffers, call this to finalize the plpeline layout.
    // Creates descriptor sets, etc.
    void finalizeLayout();

    // Add a pipeline (compute shader pass) constructed from given shader code and entry point.
    void addPipeline(const std::vector<char>& shaderCode, const char* entryPoint);
    void addPipeline(const std::vector<char>& shaderCode, const char* entryPoint, vk::SpecializationInfo*);

    // Record dispatch command after the pipeline is ready.
    void recordCommands(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

    // Signal the semaphore to kickstart for graphics -> compute submit order
    void signalSemaphore();

    // Submit the given graphics commands and the corresponding compute commands
    void submitSeqGraphicsCompute(const vk::CommandBuffer* bufferToSubmit, uint32_t currentFrame, vk::Semaphore graphicsSem);

    ~ComputeSystem();
};

