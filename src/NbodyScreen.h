#pragma once

#include "CameraScreen.h"
#include "GraphicsPipelineBuilder.h"
#include "ComputeSystem.h"


class NbodyScreen : public CameraScreen {
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

    float colorShift = 0.25f;
    float bgBrightness = 0.125f;
    float timeMultiplier = 0.05f;

public:
    const uint32_t workgroupSize = 256;
    uint32_t particleCount = maxParticleCount;
    constexpr static uint32_t particlesPerAttractor = 4096;
    constexpr static glm::vec3 attractors[] = {
        glm::vec3(5.0f, 0.0f, 0.0f),
        glm::vec3(-5.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 5.0f),
        glm::vec3(0.0f, 0.0f, -5.0f),
        glm::vec3(0.0f, 4.0f, 0.0f),
        glm::vec3(0.0f, -8.0f, 0.0f),
    };
    constexpr static uint32_t maxParticleCount = std::size(attractors) * particlesPerAttractor;

public:
    NbodyScreen(VulkanBaseApp* app);

    virtual void recordRenderCommands(vk::RenderPassBeginInfo, vk::CommandBuffer, uint32_t) override;
    virtual void submitGraphics(const vk::CommandBuffer*, uint32_t) override;
    virtual void update(float) override;

#ifdef USE_IMGUI
    virtual void imgui() override;
#endif

    virtual ~NbodyScreen();
};

