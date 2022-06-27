#ifndef MODEL_VIEWER_SCREEN_H
#define MODEL_VIEWER_SCREEN_H

#include "VulkanBaseApp.h"


struct UniformBufferObject {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 cameraPosition;
};



class ModelViewScreen : public AppScreen {
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

    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;

public:
    ModelViewScreen(VulkanBaseApp* app):
        AppScreen(app),
        model(app)
    {
        model.loadFile("../assets/FlightHelmet/FlightHelmet.gltf");
        model.createBuffers();
        createBuffers();

        createDescriptorPool();
        createDescriptorSetLayout();
        createDescriptorSets();
        createGraphicsPipeline();
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readBinaryFile("shaders/shader.vert.spv");
        auto fragShaderCode = readBinaryFile("shaders/shader.frag.spv");

        auto vertShaderModule = app->createShaderModule(vertShaderCode);
        auto fragShaderModule = app->createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo shaderStages[] = { 
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

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        auto bindingDescription = Model::vertexBindingDescription;
        auto attributeDescriptions = Model::vertexAttributeDescription;

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        vk::PipelineViewportStateCreateInfo viewportState(
                {},
                1, nullptr,
                1, nullptr // viewport and scissors are dynamic, hence nullptr (ignored)
                );

        vk::PipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizer.depthBiasEnable = VK_FALSE;

        vk::PipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = VK_FALSE;

        vk::PipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = vk::LogicOp::eCopy;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::array<vk::DescriptorSetLayout, 2> setLayouts = {
            descriptorSetLayouts.perFrame,
            descriptorSetLayouts.perMaterial,
        };
        // Push constants information
        std::array<vk::PushConstantRange, 1> pushConstants = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4)),
        };
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({},
                setLayouts.size(), setLayouts.data(),
                pushConstants.size(), pushConstants.data()
                );

        try {
            pipelineLayout = app->device->createPipelineLayout(pipelineLayoutInfo);
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // Depth testing
        vk::PipelineDepthStencilStateCreateInfo depthStencil({}, true, true, vk::CompareOp::eLess, false, false);

        vk::GraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = app->renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = nullptr;
        pipelineInfo.pDepthStencilState = &depthStencil;

        std::array<vk::DynamicState, 2> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates.size(), dynamicStates.data());
        pipelineInfo.pDynamicState = &dynamicState;

        try {
            graphicsPipeline = app->device->createGraphicsPipeline(nullptr, pipelineInfo).value;
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    /*
       Descriptor
   */

    void createDescriptorSetLayout() {
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
    }

    void createBuffers() {
        // Uniform buffers
        vk::DeviceSize uniformBufferSize = sizeof(UniformBufferObject);
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            app->createBuffer(
                    uniformBufferSize,
                    vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                    uniformBuffers[i], uniformBuffersMemory[i]
                    );
        }
    }

    void createDescriptorPool() {
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
    }

    void createDescriptorSets() {
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
            vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));

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
    }


    void updateUniformBuffer(uint32_t index) {
        glm::vec3 cameraPosition = glm::vec3(
            glm::rotate(glm::mat4(1.0f), app->time * glm::radians(90.0f), WORLD_UP) * glm::vec4(3.0f, 0.0f, 0.0f, 1.0f)
                );

        UniformBufferObject ubo {
            glm::lookAt(cameraPosition, glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP),
            glm::perspective(glm::radians(60.0f), app->swapChainExtent.width / (float) app->swapChainExtent.height, 0.1f, 10.0f),
            cameraPosition
        };
        // Y coordinate is inverted
        ubo.proj[1][1] *= -1;

        void* data = app->device->mapMemory(uniformBuffersMemory[index], 0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        app->device->unmapMemory(uniformBuffersMemory[index]);
    }

    virtual void preGraphicsSubmit(uint32_t index) override {
        updateUniformBuffer(index);
    }


    virtual void recordRenderCommands(vk::CommandBuffer commandBuffer, uint32_t index) override {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[index], 0, nullptr);

        model.render(commandBuffer, pipelineLayout,
                glm::scale(glm::mat4(1.0f), glm::vec3(3.0f))
                );
    }

    virtual ~ModelViewScreen() {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            app->device->destroyBuffer(uniformBuffers[i]);
            app->device->freeMemory(uniformBuffersMemory[i]);
        }

        app->device->destroyPipeline(graphicsPipeline);
        app->device->destroyPipelineLayout(pipelineLayout);

        // DescriptorSets are removed automatically with descriptorPool
        app->device->destroyDescriptorPool(descriptorPool);
        app->device->destroyDescriptorSetLayout(descriptorSetLayouts.perFrame);
        app->device->destroyDescriptorSetLayout(descriptorSetLayouts.perMaterial);

        model.cleanup();
    }
};


#endif
