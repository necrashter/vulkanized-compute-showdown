#include "ComputeParticleScreen.h"

struct ComputeUBO {
    alignas(16) glm::vec3 playerPos;
};

struct Particle {
    glm::vec3 pos;
    glm::vec3 vel;
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
                vk::Format::eR32G32B32Sfloat, offsetof(Particle, pos)),

        vk::VertexInputAttributeDescription(
                1, 0, // location and binding
                vk::Format::eR32G32B32Sfloat, offsetof(Particle, vel)),
    };

    float randcoor() {
        return (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    std::vector<Particle> generateShaderData(uint32_t count) {
        std::vector<Particle> particles(count);
        for (uint32_t i = 0; i < count; ++i) {
            particles[i].pos = glm::vec3(randcoor(), randcoor(), randcoor());
            particles[i].vel = glm::vec3(randcoor(), randcoor(), randcoor());
        }
        return particles;
    }
}



ComputeParticleScreen::ComputeParticleScreen(VulkanBaseApp* app):
    CameraScreen(app),
    compute(app)
{
    prepareGraphicsPipeline();
    prepareComputePipeline();
}

////////////////////////////////////////////////////////////////////////
//                         GRAPHICS PIPELINE                          //
////////////////////////////////////////////////////////////////////////


void ComputeParticleScreen::prepareGraphicsPipeline() {
    // Create Descriptor Pool
    // ---------------------------------------------------------------

    std::array<vk::DescriptorPoolSize, 1> poolSizes = {
        // UBO
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
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

    vk::DescriptorSetLayoutBinding uboBinding(
            0, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            nullptr);

    try {
        graphics.descriptorSetLayout = app->device->createDescriptorSetLayout({{}, 1, &uboBinding});
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
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorBufferInfo bufferInfo(cameraUniform.buffers[i], 0, sizeof(CameraUBO));

        std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {
            // UBO
            vk::WriteDescriptorSet(
                    graphics.descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer,
                    nullptr, // image info
                    &bufferInfo
                    ),
        };
        app->device->updateDescriptorSets(
                descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
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
    auto shaderData = generateShaderData(particleCount);

    computeSSBO = compute.createShaderStorage(shaderData.data(), sizeof(Particle) * shaderData.size());
    computeUBOmap = compute.createUniformBuffer(sizeof(ComputeUBO));
    compute.finalizeLayout();

    compute.addPipeline(readBinaryFile("shaders/particles.comp.spv"), "main");
    compute.recordCommands(particleCount / workgroupSize, 1, 1);

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
                compute.graphicsAcquireBarriers.size(), compute.graphicsAcquireBarriers.data(),
                0, nullptr);
    }

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
                compute.graphicsReleaseBarriers.size(), compute.graphicsReleaseBarriers.data(),
                0, nullptr);
    }
}


void ComputeParticleScreen::submitGraphics(const vk::CommandBuffer* bufferToSubmit, uint32_t currentFrame) {
    updateUniformBuffer(currentFrame);

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
                1, compute.getCommandBufferPointer(),
                std::size(signalSemaphores), signalSemaphores),
                // signal fence here, we're done with this frame
                app->inFlightFences[currentFrame]);
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to submit compute command buffer");
    }
}



ComputeParticleScreen::~ComputeParticleScreen() {
    // Graphics pipeline cleanup
    app->device->destroyPipeline(graphics.pipeline);
    app->device->destroyPipelineLayout(graphics.pipelineLayout);
    app->device->destroyDescriptorPool(graphics.descriptorPool);
    app->device->destroyDescriptorSetLayout(graphics.descriptorSetLayout);

    app->device->destroySemaphore(graphics.sem);
}
