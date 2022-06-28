#pragma once

#include "CameraScreen.h"
#include "GraphicsPipelineBuilder.h"
#include "ComputeSystem.h"


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

    ComputeSystem compute;
    vk::Buffer* computeSSBO;
    void* computeUBOmap;

    void prepareGraphicsPipeline();
    void prepareComputePipeline();

    uint32_t workgroupSize = 256;
    uint32_t particleCount = 256 * 1024;

public:
    ComputeParticleScreen(VulkanBaseApp* app);

    virtual void recordRenderCommands(vk::RenderPassBeginInfo, vk::CommandBuffer, uint32_t) override;
    virtual void submitGraphics(const vk::CommandBuffer*, uint32_t) override;

    virtual ~ComputeParticleScreen();
};

