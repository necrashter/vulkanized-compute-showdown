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
    CameraScreen(app)
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


////////////////////////////////////////////////////////////////////////
//                          COMPUTE PIPELINE                          //
////////////////////////////////////////////////////////////////////////

void ComputeParticleScreen::prepareComputePipeline() {
    // Create Descriptor Pool
    // ---------------------------------------------------------------

    vk::DescriptorPoolSize poolSizes[] = {
        // UBO
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1),
    };
    vk::DescriptorPoolCreateInfo poolCreateInfo({},
            1,
            std::size(poolSizes), poolSizes);
    try {
        compute.descriptorPool = app->device->createDescriptorPool(poolCreateInfo);
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    // Create Descriptor Set Layout
    // ---------------------------------------------------------------

    vk::DescriptorSetLayoutBinding binding[] = {
        vk::DescriptorSetLayoutBinding(
            0, vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eCompute,
            nullptr),
        vk::DescriptorSetLayoutBinding(
            1, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eCompute,
            nullptr),
    };

    try {
        compute.descriptorSetLayout = app->device->createDescriptorSetLayout({{},
                std::size(binding), binding});
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create Buffers
    // ---------------------------------------------------------------

    app->createBuffer(
            sizeof(ComputeUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            compute.uniformBuffer, compute.uniformBufferMemory
            );

    std::vector<Particle> shaderData = generateShaderData(particleCount);
    size_t shaderDataSize = sizeof(Particle) * shaderData.size();
    createComputeShaderStorage(shaderData.data(), shaderDataSize);

    // Create Descriptor Sets
    // ---------------------------------------------------------------

    try {
        vk::DescriptorSetLayout layouts[] = { compute.descriptorSetLayout };
        compute.descriptorSet = app->device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
                    compute.descriptorPool,
                    std::size(layouts), layouts))[0];
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    {
        vk::DescriptorBufferInfo storageBufferInfo(compute.storageBuffer, 0, shaderDataSize);
        vk::DescriptorBufferInfo uboBufferInfo(compute.uniformBuffer, 0, sizeof(ComputeUBO));

        vk::WriteDescriptorSet descriptorWrites[] = {
            // Storage
            vk::WriteDescriptorSet(
                    compute.descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer,
                    nullptr, // image info
                    &storageBufferInfo
                    ),
            // UBO
            vk::WriteDescriptorSet(
                    compute.descriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer,
                    nullptr, // image info
                    &uboBufferInfo
                    ),
        };
        app->device->updateDescriptorSets(
                std::size(descriptorWrites), descriptorWrites, 0, nullptr);
    }

    // Create Command Pool & Buffer
    // ---------------------------------------------------------------
    {
        vk::CommandPoolCreateInfo poolInfo({}, app->queueFamilyIndices.compute);
        compute.commandPool = app->device->createCommandPool(poolInfo);

        compute.commandBuffer = app->device->allocateCommandBuffers(
                vk::CommandBufferAllocateInfo(
                    compute.commandPool,
                    vk::CommandBufferLevel::ePrimary,
                    1
                    ))[0];
    }

    // Create Pipeline
    // ---------------------------------------------------------------

    auto shaderModule = app->createShaderModule(readBinaryFile("shaders/particles.comp.spv"));

    compute.pipelineLayout = app->device->createPipelineLayout(
            vk::PipelineLayoutCreateInfo({}, compute.descriptorSetLayout));

    vk::ComputePipelineCreateInfo ComputePipelineCreateInfo(
            vk::PipelineCreateFlags(),    // Flags
            vk::PipelineShaderStageCreateInfo(       // Shader Create Info struct
                    vk::PipelineShaderStageCreateFlags(),  // Flags
                    vk::ShaderStageFlagBits::eCompute,     // Stage
                    shaderModule.get(),                    // Shader Module
                    "main"                                 // Shader Entry Point
                    ),
            compute.pipelineLayout                // Pipeline Layout
            );
    compute.pipeline = app->device->createComputePipeline(nullptr, ComputePipelineCreateInfo).value;

    // record
    recordComputeCommandBuffer(shaderData.size() / workgroupSize, 1, 1);

    // Semaphore

    compute.sem = app->device->createSemaphore({});
    // Signal the semaphore (kickstart)
    vk::SubmitInfo signalSubmit(
            0, nullptr, nullptr, // wait nothing
            0, nullptr, // do nothing
            1, &compute.sem); // just signal
    app->computeQueue.submit(signalSubmit);
}



void ComputeParticleScreen::createComputeShaderStorage(const void* input, size_t size) {
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    app->createBuffer(
            size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory);

    void* data = app->device->mapMemory(stagingBufferMemory, 0, size);
    memcpy(data, input, size);
    app->device->unmapMemory(stagingBufferMemory);

    app->createBuffer(
            size,
            // NOTE: compute shader storageBuffer is also used as vertex buffer
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            compute.storageBuffer, compute.storageBufferMemory);
    app->copyBuffer(stagingBuffer, compute.storageBuffer, size);

    app->device->destroyBuffer(stagingBuffer);
    app->device->freeMemory(stagingBufferMemory);
}



void ComputeParticleScreen::recordComputeCommandBuffer(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    uint32_t graphicsQindex = app->queueFamilyIndices.graphics;
    uint32_t computeQindex = app->queueFamilyIndices.compute;

    // for some reason we need simultaneous use
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
    if (compute.commandBuffer.begin(&beginInfo) != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to begin recording to compute command buffer");
    }

    // TODO: apparently you need to do this after initialization too
    if (graphicsQindex != computeQindex) {
        // need memory barrier if graphics and compute queues are different
        // Make sure that graphics pipeline red the previous inputs
        vk::BufferMemoryBarrier barrier(
                // source mask, destination mask
                {}, vk::AccessFlagBits::eShaderWrite,
                // source and destination queues
                graphicsQindex, computeQindex,
                // buffer range
                compute.storageBuffer, 0, particleCount * sizeof(Particle)
                );
        compute.commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eVertexInput,
                vk::PipelineStageFlagBits::eComputeShader,
                {}, // dependency flags
                0, nullptr, 1, &barrier, 0, nullptr);
    }

    compute.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, compute.pipeline);
    compute.commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,    // Bind point
            compute.pipelineLayout,             // Pipeline Layout
            0,                                  // First descriptor set
            { compute.descriptorSet },          // List of descriptor sets
            {});                                // Dynamic offsets
    compute.commandBuffer.dispatch(groups_x, groups_y, groups_z);

    if (graphicsQindex != computeQindex) {
        // Make sure that graphics pipeline doesn't read incomplete data
        vk::BufferMemoryBarrier barrier(
                // source mask, destination mask
                vk::AccessFlagBits::eShaderWrite, {},
                // source and destination queues
                computeQindex, graphicsQindex,
                // buffer range
                compute.storageBuffer, 0, particleCount * sizeof(Particle)
                );
        compute.commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eVertexInput,
                {}, // dependency flags
                0, nullptr, 1, &barrier, 0, nullptr);
    }

    compute.commandBuffer.end();
}



void ComputeParticleScreen::recordRenderCommands(vk::CommandBuffer commandBuffer, uint32_t index) {
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, 1, &graphics.descriptorSets[index], 0, nullptr);
    vk::DeviceSize offsets[] = {0};
    commandBuffer.bindVertexBuffers(0, 1, &compute.storageBuffer, offsets);
    commandBuffer.draw(particleCount, 1, 0, 0);
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
                1, &compute.commandBuffer,
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

    // Compute pipeline cleanup
    app->device->destroyPipeline(compute.pipeline);
    app->device->destroyPipelineLayout(compute.pipelineLayout);
    app->device->destroyDescriptorPool(compute.descriptorPool);
    app->device->destroyDescriptorSetLayout(compute.descriptorSetLayout);
    app->device->destroyCommandPool(compute.commandPool);

    app->device->destroyBuffer(compute.storageBuffer);
    app->device->freeMemory(compute.storageBufferMemory);
    app->device->destroyBuffer(compute.uniformBuffer);
    app->device->freeMemory(compute.uniformBufferMemory);

    app->device->destroySemaphore(graphics.sem);
    app->device->destroySemaphore(compute.sem);
}
