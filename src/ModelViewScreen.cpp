#include "ModelViewScreen.h"

ModelViewScreen::ModelViewScreen(VulkanBaseApp* app):
    CameraScreen(app),
    model(app)
{
    model.addVertexAttribute("POSITION", vk::Format::eR32G32B32Sfloat);
    model.addVertexAttribute("NORMAL", vk::Format::eR32G32B32Sfloat);
    model.addVertexAttribute("TEXCOORD_0", vk::Format::eR32G32Sfloat);
    model.loadFile("../assets/VikingRoom.gltf");
    model.createBuffers();

    noclipCam.position = glm::vec3(6.5, 6.9, 6.9);
    noclipCam.pitch = -35;
    noclipCam.yaw = -130.0;
    noclipCam.update_vectors();

    // Create Descriptor Pool
    // ---------------------------------------------------------------

    size_t materialCount = model.materials.size();
    std::array<vk::DescriptorPoolSize, 2> poolSizes = {
        // UBO
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
        // Sampler
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, materialCount),
    };
    vk::DescriptorPoolCreateInfo poolCreateInfo({},
            MAX_FRAMES_IN_FLIGHT + materialCount,
            poolSizes.size(), poolSizes.data());
    try {
        descriptorPool = app->device->createDescriptorPool(poolCreateInfo);
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    // Create Descriptor Set Layout
    // ---------------------------------------------------------------

    vk::DescriptorSetLayoutBinding uboBinding(
            0, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            nullptr);

    try {
        descriptorSetLayouts.perFrame = app->device->createDescriptorSetLayout({{}, 1, &uboBinding});
        descriptorSetLayouts.perMaterial = model.createMaterialDescriptorSetLayout();
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    // Create Descriptor Sets
    // ---------------------------------------------------------------

    try {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayouts.perFrame);
        descriptorSets = app->device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
                    descriptorPool,
                    static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
                    layouts.data()));
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorBufferInfo bufferInfo(cameraUniform.buffers[i], 0, sizeof(CameraUBO));

        std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {
            // UBO
            vk::WriteDescriptorSet(
                    descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer,
                    nullptr, // image info
                    &bufferInfo
                    ),
        };
        app->device->updateDescriptorSets(
                descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    model.createMaterialDescriptorSets(descriptorPool, descriptorSetLayouts.perMaterial);


    // Create Graphics Pipeline
    // ---------------------------------------------------------------

    GraphicsPipelineBuilder pipelineBuilder;

    auto vertShaderModule = app->createShaderModule(readBinaryFile("shaders/shader.vert.spv"));
    auto fragShaderModule = app->createShaderModule(readBinaryFile("shaders/shader.frag.spv"));

    pipelineBuilder.stages = { 
        {
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eVertex,
            *vertShaderModule,
            "main"
        }, 
        {
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eFragment,
            *fragShaderModule,
            "main"
        } 
    };

    auto bindingDescription = model.getVertexInputBindingDescription();
    auto attributeDescriptions = model.getVertexAttributeDescriptions();

    pipelineBuilder.vertexInput.vertexBindingDescriptionCount = 1;
    pipelineBuilder.vertexInput.pVertexBindingDescriptions = &bindingDescription;
    pipelineBuilder.vertexInput.vertexAttributeDescriptionCount = attributeDescriptions.size();
    pipelineBuilder.vertexInput.pVertexAttributeDescriptions = attributeDescriptions.data();


    pipelineBuilder.descriptorSetLayouts = {
        descriptorSetLayouts.perFrame,
        descriptorSetLayouts.perMaterial,
    };
    pipelineBuilder.pushConstants = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4)),
    };

    pipelineBuilder.pipelineInfo.renderPass = app->renderPass;

    try {
        graphicsPipeline = pipelineBuilder.build(app->device.get());
        pipelineLayout = pipelineBuilder.pipelineLayout;
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }
}


void ModelViewScreen::recordRenderCommands(vk::RenderPassBeginInfo renderPassInfo, vk::CommandBuffer commandBuffer, uint32_t index) {
    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[index], 0, nullptr);

    model.render(commandBuffer, pipelineLayout,
            glm::scale(glm::mat4(1.0f), glm::vec3(3.0f))
            );

    app->renderUI(commandBuffer);
    commandBuffer.endRenderPass();
}


ModelViewScreen::~ModelViewScreen() {
    app->device->destroyPipeline(graphicsPipeline);
    app->device->destroyPipelineLayout(pipelineLayout);

    // DescriptorSets are removed automatically with descriptorPool
    app->device->destroyDescriptorPool(descriptorPool);
    app->device->destroyDescriptorSetLayout(descriptorSetLayouts.perFrame);
    app->device->destroyDescriptorSetLayout(descriptorSetLayouts.perMaterial);

    model.cleanup();
}
