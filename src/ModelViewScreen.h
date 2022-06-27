#ifndef MODEL_VIEWER_SCREEN_H
#define MODEL_VIEWER_SCREEN_H

#include "VulkanBaseApp.h"
#include "GraphicsPipelineBuilder.h"
#include "Noclip.h"


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

    noclip::cam noclipCam;

public:
    ModelViewScreen(VulkanBaseApp* app):
        AppScreen(app),
        model(app),
        noclipCam(app->window)
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

        auto bindingDescription = Model::vertexBindingDescription;
        auto attributeDescriptions = Model::vertexAttributeDescription;

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
        // glm::vec3 cameraPosition = glm::vec3(
        //     glm::rotate(glm::mat4(1.0f), app->time * glm::radians(90.0f), WORLD_UP) * glm::vec4(3.0f, 0.0f, 0.0f, 1.0f)
        //         );

        UniformBufferObject ubo {
            // glm::lookAt(cameraPosition, glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP),
            noclipCam.get_view_matrix(),
            glm::perspective(glm::radians(60.0f), app->swapChainExtent.width / (float) app->swapChainExtent.height, 0.1f, 10.0f),
            noclipCam.position
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

    virtual void mouseMovementCallback(GLFWwindow* window, double xpos, double ypos) override {
        static double lastxpos = 0, lastypos = 0;

        double xdiff = xpos - lastxpos;
        double ydiff = lastypos - ypos;

#ifdef USE_IMGUI
        if (!ImGui::IsAnyItemActive())
#endif
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            noclipCam.process_mouse(xdiff, ydiff);
            noclipCam.update_vectors();
        }

        lastxpos = xpos;
        lastypos = ypos;
    }

    virtual void update(float delta) override {
        noclipCam.update(delta);
    }

#ifdef USE_IMGUI
    virtual void imgui() override {
    }
#endif

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
