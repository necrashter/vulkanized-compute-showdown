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
    FrameUniform* computeUBO;

    FrameUniform graphicsUniform;

    void prepareGraphicsPipeline();
    void prepareComputePipeline();

    const uint32_t workgroupSize = 256;
    uint32_t particleCount = 125000;
    const uint32_t maxParticleCount = 256 * 1024;

    float colorShift = 0.25f;
    float particleRange = 5.0f;
    float bgBrightness = 0.125f;
    float baseSpeed = 1.0f;
    float speedVariation = 1.0f;
    float restartParticles = false;

public:
    ComputeParticleScreen(VulkanBaseApp* app);

    virtual void recordRenderCommands(vk::RenderPassBeginInfo, vk::CommandBuffer, uint32_t) override;
    virtual void submitGraphics(const vk::CommandBuffer*, uint32_t) override;
    virtual void update(float) override;

#ifdef USE_IMGUI
    virtual void imgui() override;
#endif

    virtual ~ComputeParticleScreen();
};

