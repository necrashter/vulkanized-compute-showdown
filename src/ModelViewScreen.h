#ifndef MODEL_VIEWER_SCREEN_H
#define MODEL_VIEWER_SCREEN_H

#include "CameraScreen.h"
#include "GraphicsPipelineBuilder.h"
#include "Model.h"


class ModelViewScreen : public CameraScreen {
private:
    Model model;

    struct {
        vk::DescriptorSetLayout perFrame;
        vk::DescriptorSetLayout perMaterial;
    } descriptorSetLayouts;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

public:
    ModelViewScreen(VulkanBaseApp* app);

    virtual void recordRenderCommands(vk::RenderPassBeginInfo, vk::CommandBuffer, uint32_t) override;

    virtual ~ModelViewScreen();
};


#endif
