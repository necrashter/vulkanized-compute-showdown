#pragma once

#include "CameraScreen.h"
#include "GraphicsPipelineBuilder.h"
#include "ComputeSystem.h"

#include "Model.h"
#ifdef USE_LIBKTX
#include "Texture.h"
#endif


class RigidScreen : public CameraScreen {
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
    std::vector<Primitive> primitives;
    int selectedPrimitiveIndex;
    Model model;

    void buildPipeline(void* oldData=nullptr, size_t oldDataSize=0);
    void prepareGraphicsPipeline();
    void prepareComputePipeline(void* oldData=nullptr, size_t oldDataSize=0);
    void pipelineCleanup();

    float colorShift = 0.25f;
    float bgBrightness = 0.125f;
    float timeMultiplier = 1.0f;
    float gravity = 9.8f;

    uint32_t workgroupSize = 256;

public:
    uint32_t particleCount;
    uint32_t selectedParticles = 1024*4;
    constexpr static uint32_t maxParticles = 1024 * 16;

public:
    RigidScreen(VulkanBaseApp* app);

    virtual void recordRenderCommands(vk::RenderPassBeginInfo, vk::CommandBuffer, uint32_t) override;
    virtual void submitGraphics(const vk::CommandBuffer*, uint32_t) override;
    virtual void update(float) override;

#ifdef USE_IMGUI
    virtual void imgui() override;
#endif

    virtual ~RigidScreen();
};

