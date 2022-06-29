#include "ComputeParticleScreen.h"
#include "glsl.h"

struct ComputeUBO {
    alignas(4) glm::uint32 particleCount;
    alignas(4) glm::float32 delta;
    alignas(4) glm::float32 range2;
    alignas(4) glm::float32 time;
    alignas(4) glm::float32 baseSpeed;
    alignas(4) glm::float32 speedVariation;
    alignas(4) glm::uint32 restart;
};

struct FrameUBO {
    alignas(4) glm::float32 colorShift;
};

struct Particle {
    alignas(16) glm::vec4 pos;
    alignas(16) glm::vec3 vel;
    alignas(4)  glm::float32 color;
};


namespace {
    constexpr vk::VertexInputBindingDescription vertexBindingDescription = {
        0, // binding
        sizeof(Particle), // stride
        vk::VertexInputRate::eVertex,
    };
    constexpr vk::VertexInputAttributeDescription vertexAttributeDescriptions[] = {
        vk::VertexInputAttributeDescription(
                0, 0, // location and binding
                vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, pos)),

        vk::VertexInputAttributeDescription(
                1, 0, // location and binding
                vk::Format::eR32G32B32Sfloat, offsetof(Particle, vel)),

        vk::VertexInputAttributeDescription(
                2, 0, // location and binding
                vk::Format::eR32Sfloat, offsetof(Particle, color)),
    };

    float randcoor() {
        return (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    std::vector<Particle> generateShaderData(uint32_t count) {
        srand((unsigned) time(NULL));
        std::vector<Particle> particles(count);
        for (uint32_t i = 0; i < count; ++i) {
            particles[i].pos = glm::vec4(
                    0, 0, 0,
                    rand()/(float)RAND_MAX * 4.0f + 4.0f
                    );
            particles[i].vel = glm::normalize(glm::vec3(randcoor(), randcoor(), randcoor()));
            particles[i].color = rand()/(float)RAND_MAX * 0.25;
        }
        return particles;
    }
}



ComputeParticleScreen::ComputeParticleScreen(VulkanBaseApp* app):
    CameraScreen(app),
    compute(app),
    graphicsUniform(app, sizeof(FrameUBO))
{
    prepareGraphicsPipeline();
    prepareComputePipeline();
    noclipCam.position = glm::vec3(-6*1.7320508075688772, 6, 0);
    noclipCam.pitch = -30;
    noclipCam.yaw = 0.0;
    noclipCam.update_vectors();
}

////////////////////////////////////////////////////////////////////////
//                         GRAPHICS PIPELINE                          //
////////////////////////////////////////////////////////////////////////


void ComputeParticleScreen::prepareGraphicsPipeline() {
    // Create Descriptor Pool
    // ---------------------------------------------------------------

    std::array<vk::DescriptorPoolSize, 1> poolSizes = {
        // UBO
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT * 2),
    };
    vk::DescriptorPoolCreateInfo poolCreateInfo({},
            MAX_FRAMES_IN_FLIGHT,
            poolSizes.size(), poolSizes.data());
    try {
        graphics.descriptorPool = app->device->createDescriptorPool(poolCreateInfo);
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    // Create Descriptor Set Layout
    // ---------------------------------------------------------------

    vk::DescriptorSetLayoutBinding bindings[] = {
        vk::DescriptorSetLayoutBinding(
            0, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            nullptr),
        vk::DescriptorSetLayoutBinding(
            1, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            nullptr),
    };

    try {
        graphics.descriptorSetLayout = app->device->createDescriptorSetLayout({{}, std::size(bindings), bindings});
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create Descriptor Sets
    // ---------------------------------------------------------------

    try {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, graphics.descriptorSetLayout);
        graphics.descriptorSets = app->device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
                    graphics.descriptorPool,
                    static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
                    layouts.data()));
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorBufferInfo bufferInfo(cameraUniform.buffers[i], 0, sizeof(CameraUBO));
        vk::DescriptorBufferInfo frameUniformInfo(graphicsUniform.buffers[i], 0, sizeof(FrameUBO));

        vk::WriteDescriptorSet descriptorWrites[] = {
            // UBO
            vk::WriteDescriptorSet(
                    graphics.descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer,
                    nullptr, // image info
                    &bufferInfo
                    ),
            vk::WriteDescriptorSet(
                    graphics.descriptorSets[i], 1, 0, 1, vk::DescriptorType::eUniformBuffer,
                    nullptr, // image info
                    &frameUniformInfo
                    ),
        };
        app->device->updateDescriptorSets(std::size(descriptorWrites), descriptorWrites, 0, nullptr);
    }

    // Create Graphics Pipeline
    // ---------------------------------------------------------------

    GraphicsPipelineBuilder pipelineBuilder;

    auto vertShaderModule = app->createShaderModule(readBinaryFile("shaders/particles.vert.spv"));
    auto fragShaderModule = app->createShaderModule(readBinaryFile("shaders/particles.frag.spv"));

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

    // Additive blend
    pipelineBuilder.colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    pipelineBuilder.colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    pipelineBuilder.colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOne;
    pipelineBuilder.colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
    pipelineBuilder.colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    pipelineBuilder.colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eOne;
    // No depth test
    pipelineBuilder.depthStencil.depthTestEnable = VK_FALSE;
    pipelineBuilder.depthStencil.depthWriteEnable = VK_FALSE;

    pipelineBuilder.rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    pipelineBuilder.inputAssembly.topology = vk::PrimitiveTopology::ePointList;

    pipelineBuilder.vertexInput.vertexBindingDescriptionCount = 1;
    pipelineBuilder.vertexInput.pVertexBindingDescriptions = &vertexBindingDescription;
    pipelineBuilder.vertexInput.vertexAttributeDescriptionCount = std::size(vertexAttributeDescriptions);
    pipelineBuilder.vertexInput.pVertexAttributeDescriptions = vertexAttributeDescriptions;

    pipelineBuilder.descriptorSetLayouts = { graphics.descriptorSetLayout };

    pipelineBuilder.pipelineInfo.renderPass = app->renderPass;

    try {
        graphics.pipeline = pipelineBuilder.build(app->device.get());
        graphics.pipelineLayout = pipelineBuilder.pipelineLayout;
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }


    // Semaphore
    graphics.sem = app->device->createSemaphore({});
}



void ComputeParticleScreen::prepareComputePipeline() {
    auto shaderData = generateShaderData(maxParticleCount);

    computeSSBO = compute.createShaderStorage(shaderData.data(), sizeof(Particle) * shaderData.size());
    computeUBO = compute.createUniformBuffer(sizeof(ComputeUBO));
    compute.finalizeLayout();

    compute.addPipeline(readBinaryFile("shaders/particles.comp.spv"), "main");
    compute.recordCommands(maxParticleCount / workgroupSize, 1, 1);

    // kickstart
    compute.signalSemaphore();
}




void ComputeParticleScreen::recordRenderCommands(vk::RenderPassBeginInfo renderPassInfo, vk::CommandBuffer commandBuffer, uint32_t index) {
    uint32_t graphicsQindex = app->queueFamilyIndices.graphics;
    uint32_t computeQindex = app->queueFamilyIndices.compute;

    // Compute shader barrier
    if (graphicsQindex != computeQindex) {
        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eVertexInput,
                {},
                0, nullptr,
                compute.graphicsAcquireBarriers[index].size(), compute.graphicsAcquireBarriers[index].data(),
                0, nullptr);
    }

    glm::vec3 bgColor = glsl::hsv2rgb(glm::vec3(colorShift, 0.25f, bgBrightness));
    vk::ClearValue clearValues[] = {
        vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{
                    bgColor.r, bgColor.r, bgColor.b,
                    1.0f })),
            vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0)),
    };
    renderPassInfo.pClearValues = clearValues;
    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, 1, &graphics.descriptorSets[index], 0, nullptr);
    vk::DeviceSize offsets[] = {0};
    commandBuffer.bindVertexBuffers(0, 1, computeSSBO, offsets);
    commandBuffer.draw(particleCount, 1, 0, 0);

    app->renderUI(commandBuffer);
    commandBuffer.endRenderPass();

    // Release barrier
    if (graphicsQindex != computeQindex) {
        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eVertexInput,
                vk::PipelineStageFlagBits::eComputeShader,
                {},
                0, nullptr,
                compute.graphicsReleaseBarriers[index].size(), compute.graphicsReleaseBarriers[index].data(),
                0, nullptr);
    }
}


void ComputeParticleScreen::update(float delta) {
    CameraScreen::update(delta);
}


void ComputeParticleScreen::submitGraphics(const vk::CommandBuffer* bufferToSubmit, uint32_t currentFrame) {
    updateUniformBuffer(currentFrame);
    {
        ComputeUBO* computeUbo = (ComputeUBO*)computeUBO->mappings[currentFrame];
        computeUbo->particleCount = particleCount;
        computeUbo->delta = app->delta;
        computeUbo->range2 = particleRange * particleRange;
        computeUbo->time = app->time;
        computeUbo->baseSpeed = baseSpeed;
        computeUbo->speedVariation = speedVariation;
        computeUbo->restart = restartParticles;
        restartParticles = false;

        FrameUBO* ubo = (FrameUBO*)graphicsUniform.mappings[currentFrame];
        ubo->colorShift = colorShift;
    }

    try {
        // Wait for compute shader to complete at vertex input stage (see kickstart)
        // Wait for output image to become available at color attachment output stage
        vk::Semaphore waitSemaphores[] = {
            compute.sem,
            app->imageAvailableSemaphores[currentFrame]
        };
        vk::PipelineStageFlags waitStages[] = {
            vk::PipelineStageFlagBits::eVertexInput,
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        vk::Semaphore signalSemaphores[] = {
            graphics.sem,
            app->renderFinishedSemaphores[currentFrame]
        };

        app->graphicsQueue.submit(vk::SubmitInfo(
                std::size(waitSemaphores), waitSemaphores, waitStages,
                1, bufferToSubmit,
                std::size(signalSemaphores), signalSemaphores));
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to submit graphics command buffer");
    }

    try {
        // Wait for graphics queue to render
        vk::Semaphore waitSemaphores[] = {
            graphics.sem,
        };
        vk::PipelineStageFlags waitStages[] = {
            vk::PipelineStageFlagBits::eComputeShader
        };
        vk::Semaphore signalSemaphores[] = {
            compute.sem,
        };

        app->computeQueue.submit(vk::SubmitInfo(
                std::size(waitSemaphores), waitSemaphores, waitStages,
                1, compute.getCommandBufferPointer(currentFrame),
                std::size(signalSemaphores), signalSemaphores),
                // signal fence here, we're done with this frame
                app->inFlightFences[currentFrame]);
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to submit compute command buffer");
    }
}


#ifdef USE_IMGUI
void ComputeParticleScreen::imgui() {
    static bool showParticleSettings = true;
    CameraScreen::imgui();
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Particle Settings", NULL, &showParticleSettings);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (showParticleSettings) {
        ImGui::Begin("Particles", &showParticleSettings);

        ImGui::DragFloat("BG Brightness", &bgBrightness, 0.005f, 0.0f, 1.0f, "%.3f");
        ImGui::DragFloat("Color Shift", &colorShift, 0.005f, 0.0f, 1.0f, "%.3f");

        ImGui::Separator();

        ImGui::DragFloat("Base Speed", &baseSpeed, 0.005f, 0.0f, 10.0f, "%.3f");
        ImGui::DragFloat("Speed Variation", &speedVariation, 0.005f, 0.0f, 10.0f, "%.3f");
        ImGui::DragFloat("Range", &particleRange, 0.25f, 1.0f, 100.0f, "%.3f");

        ImGui::Separator();

        ImGui::DragInt("Particle Count", (int*) &particleCount, 100, 1, maxParticleCount);
        if (ImGui::Button("Restart")) restartParticles = true;
        ImGui::End();
    }
}
#endif

ComputeParticleScreen::~ComputeParticleScreen() {
    // Graphics pipeline cleanup
    app->device->destroyPipeline(graphics.pipeline);
    app->device->destroyPipelineLayout(graphics.pipelineLayout);
    app->device->destroyDescriptorPool(graphics.descriptorPool);
    app->device->destroyDescriptorSetLayout(graphics.descriptorSetLayout);

    app->device->destroySemaphore(graphics.sem);
}

