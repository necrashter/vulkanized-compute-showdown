#pragma once

#include "CameraScreen.h"
#include "GraphicsPipelineBuilder.h"


class ComputeParticleScreen : public CameraScreen {
private:
    struct {
        vk::DescriptorSetLayout descriptorSetLayout;
        vk::DescriptorPool descriptorPool;
        std::vector<vk::DescriptorSet> descriptorSets;
        vk::PipelineLayout pipelineLayout;
        vk::Pipeline pipeline;
        vk::Semaphore sem;
    } graphics;

    struct {
        vk::DescriptorSetLayout descriptorSetLayout;
        vk::DescriptorPool descriptorPool;
        vk::DescriptorSet descriptorSet;
        vk::PipelineLayout pipelineLayout;
        vk::Pipeline pipeline;
        vk::Semaphore sem;
        // Need special command buffer and hence command pool
        vk::CommandPool commandPool;
        vk::CommandBuffer commandBuffer;
        // Shader storage buffer object
        vk::Buffer storageBuffer;
        vk::DeviceMemory storageBufferMemory;
        // Uniform buffer
        vk::Buffer uniformBuffer;
        vk::DeviceMemory uniformBufferMemory;
    } compute;

    void prepareGraphicsPipeline();
    void prepareComputePipeline();
    void createComputeShaderStorage(const void* data, size_t size);
    void recordComputeCommandBuffer(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

    uint32_t workgroupSize = 256;
    uint32_t particleCount = 256 * 1024;

public:
    ComputeParticleScreen(VulkanBaseApp* app);

    virtual void recordRenderCommands(vk::CommandBuffer commandBuffer, uint32_t index) override;
    virtual void submitGraphics(const vk::CommandBuffer*, uint32_t) override;

    virtual ~ComputeParticleScreen();
};

