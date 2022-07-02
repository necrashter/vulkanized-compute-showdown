#pragma once

#include "CameraScreen.h"
#include "GraphicsPipelineBuilder.h"
#include "ComputeSystem.h"
#include "Model.h"

#ifdef USE_LIBKTX
#include "Texture.h"
#endif


class NbodyRigidScreen : public CameraScreen {
private:
    struct {
        vk::DescriptorSetLayout descriptorSetLayout;
        vk::DescriptorPool descriptorPool;
        std::vector<vk::DescriptorSet> descriptorSets;
        vk::PipelineLayout pipelineLayout;
        vk::Pipeline pipeline;
        vk::Semaphore sem;
    } graphics;

    ComputeSystem* compute;
    // NOTE: These resources belong to compute
    vk::Buffer* computeSSBO;
    FrameUniform* computeUBO;

    // This belongs to us
    FrameUniform* graphicsUniform;

#ifdef USE_LIBKTX
    TextureKTX huesTexture;
    TextureKTX particleTexture;
#endif
    struct Primitive {
        uint32_t firstIndex;
        uint32_t indexCount;
    };
    Primitive planetPrimitive;
    Model model;

    void buildPipeline();
    void prepareGraphicsPipeline();
    void prepareComputePipeline();
    void pipelineCleanup();

    float colorShift = 0.25f;
    float bgBrightness = 0.125f;
    float timeMultiplier = 0.05f;

    float gravity = 0.002f;
    float power = 0.75f;
    float soften = 0.05f;
    uint32_t workgroupSize = 256;

public:
    uint32_t particleCount = maxParticleCount/4;
    constexpr static uint32_t particlesPerAttractor = 2 * 1024;
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
    NbodyRigidScreen(VulkanBaseApp* app);

    virtual void recordRenderCommands(vk::RenderPassBeginInfo, vk::CommandBuffer, uint32_t) override;
    virtual void submitGraphics(const vk::CommandBuffer*, uint32_t) override;
    virtual void update(float) override;

#ifdef USE_IMGUI
    virtual void imgui() override;
#endif

    virtual ~NbodyRigidScreen();
};

