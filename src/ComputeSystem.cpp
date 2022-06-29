#include "ComputeSystem.h"


ComputeSystem::ComputeSystem(VulkanContext* context): context(context) {
    sem = context->device->createSemaphore({});
}


////////////////////////////////////////////////////////////////////////
//                              STORAGE                               //
////////////////////////////////////////////////////////////////////////


vk::Buffer* ComputeSystem::createShaderStorage(const void* input, size_t size) {
    if (size == 0) {
        throw std::runtime_error("Compute Shader Storage Buffer size is 0");
    }
    storageBuffers.emplace_back();
    auto& storage = storageBuffers.back();

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    context->createBuffer(
            size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory);

    void* data = context->device->mapMemory(stagingBufferMemory, 0, size);
    memcpy(data, input, size);
    context->device->unmapMemory(stagingBufferMemory);

    context->createBuffer(
            size,
            // NOTE: compute shader storageBuffer is also used as vertex buffer
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            storage.buffer, storage.memory);
    context->copyBuffer(stagingBuffer, storage.buffer, size);

    context->device->destroyBuffer(stagingBuffer);
    context->device->freeMemory(stagingBufferMemory);

    storage.size = size;

    uint32_t bindingIndex = (uint32_t) bindings.size();
    bindings.emplace_back(
            bindingIndex,
            vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eCompute,
            nullptr
            );
    descriptors.push_back(&storage);

    return &storage.buffer;
}


FrameUniform* ComputeSystem::createUniformBuffer(size_t size) {
    if (size == 0) {
        throw std::runtime_error("Compute Shader Uniform Buffer size is 0");
    }
    uniformBuffers.emplace_back(context, size);
    auto& storage = uniformBuffers.back();

    uint32_t bindingIndex = (uint32_t) bindings.size();
    bindings.emplace_back(
            bindingIndex,
            vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eCompute,
            nullptr
            );
    descriptors.push_back(&storage);

    return &storage;
}


////////////////////////////////////////////////////////////////////////
//                          COMPUTE PIPELINE                          //
////////////////////////////////////////////////////////////////////////

void ComputeSystem::finalizeLayout() {
    // Create Descriptor Pool
    // ---------------------------------------------------------------

    vk::DescriptorPoolSize poolSizes[] = {
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT * uniformBuffers.size()),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * storageBuffers.size()),
    };
    vk::DescriptorPoolCreateInfo poolCreateInfo({},
            MAX_FRAMES_IN_FLIGHT,
            std::size(poolSizes), poolSizes);
    try {
        descriptorPool = context->device->createDescriptorPool(poolCreateInfo);
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    // Create Descriptor Set Layout
    // ---------------------------------------------------------------

    try {
        descriptorSetLayout = context->device->createDescriptorSetLayout({{},
                (uint32_t)bindings.size(), bindings.data()});
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create Descriptor Sets
    // ---------------------------------------------------------------

    try {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        descriptorSets = context->device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
                    descriptorPool,
                    layouts.size(), layouts.data()));
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {
        std::vector<vk::DescriptorBufferInfo> bufferInfos(
                descriptors.size()
                );
        std::vector<vk::WriteDescriptorSet> descriptorWrites(
                descriptors.size(),
                vk::WriteDescriptorSet(
                        descriptorSets[frame], 0, 0, 1, {},
                        nullptr, nullptr
                        )
                );
        for (uint32_t i = 0; i < descriptors.size(); ++i) {
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
            if (descriptors[i]->type == Descriptor::eFrameUniform) {
                descriptorWrites[i].descriptorType = vk::DescriptorType::eUniformBuffer;
                bufferInfos[i].buffer = ((FrameUniform*)descriptors[i])->buffers[frame];
                bufferInfos[i].range = ((FrameUniform*)descriptors[i])->size;
            }
            else if (descriptors[i]->type == Descriptor::eSSBO) {
                descriptorWrites[i].descriptorType = vk::DescriptorType::eStorageBuffer;
                bufferInfos[i].buffer = ((ComputeStorage*)descriptors[i])->buffer;
                bufferInfos[i].range = ((ComputeStorage*)descriptors[i])->size;
            }
        }
        context->device->updateDescriptorSets(
                descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    // Create Pipeline Layout
    // ---------------------------------------------------------------

    pipelineLayout = context->device->createPipelineLayout(
            vk::PipelineLayoutCreateInfo({}, descriptorSetLayout));
}



void ComputeSystem::addPipeline(const std::vector<char>& shaderCode, const char* entryPoint) {
    auto shaderModule = context->createShaderModule(shaderCode);

    vk::ComputePipelineCreateInfo ComputePipelineCreateInfo(
            vk::PipelineCreateFlags(),    // Flags
            vk::PipelineShaderStageCreateInfo(       // Shader Create Info struct
                    vk::PipelineShaderStageCreateFlags(),  // Flags
                    vk::ShaderStageFlagBits::eCompute,     // Stage
                    shaderModule.get(),                    // Shader Module
                    entryPoint // Shader Entry Point
                    ),
            pipelineLayout                // Pipeline Layout
            );

    pipelines.push_back(context->device->createComputePipeline(nullptr, ComputePipelineCreateInfo).value);
}


std::vector<vk::BufferMemoryBarrier> ComputeSystem::getMemoryBarriers(vk::BufferMemoryBarrier temp, uint32_t frame) {
    std::vector<vk::BufferMemoryBarrier> barriers(descriptors.size(), temp);
    for (uint32_t i = 0; i < descriptors.size(); ++i) {
        if (descriptors[i]->type == Descriptor::eSSBO) {
            auto obj = ((ComputeStorage*)descriptors[i]);
            barriers[i].buffer = obj->buffer;
            barriers[i].size = obj->size;
        }
        else if (descriptors[i]->type == Descriptor::eFrameUniform) {
            auto obj = ((FrameUniform*)descriptors[i]);
            barriers[i].buffer = obj->buffers[frame];
            barriers[i].size = obj->size;
        }
    }
    return barriers;
}


void ComputeSystem::recordCommands(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    // Create Command Pool & Buffer
    {
        vk::CommandPoolCreateInfo poolInfo({}, context->queueFamilyIndices.compute);
        commandPool = context->device->createCommandPool(poolInfo);

        commandBuffers = context->device->allocateCommandBuffers(
                vk::CommandBufferAllocateInfo(
                    commandPool,
                    vk::CommandBufferLevel::ePrimary,
                    MAX_FRAMES_IN_FLIGHT
                    ));
    }

    uint32_t graphicsQindex = context->queueFamilyIndices.graphics;
    uint32_t computeQindex = context->queueFamilyIndices.compute;

    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {
        vk::CommandBuffer commandBuffer = commandBuffers[frame];
        // for some reason we need simultaneous use
        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
        if (commandBuffer.begin(&beginInfo) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to begin recording to compute command buffer");
        }

        if (graphicsQindex != computeQindex) {
            // need memory barrier if graphics and compute queues are different
            // Make sure that graphics pipeline red the previous inputs
            // Matching transfer acquire & release barriers must be present in the graphics pipeline
            std::vector<vk::BufferMemoryBarrier> barriers = getMemoryBarriers(
                    vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        {}, vk::AccessFlagBits::eShaderWrite,
                        // source and destination queues
                        graphicsQindex, computeQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        ),
                    frame
                    );
            commandBuffer.pipelineBarrier(
                    // vertex input is not supported by dedicated compute queue families
                    vk::PipelineStageFlagBits::eDrawIndirect,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {}, // dependency flags
                    0, nullptr, barriers.size(), barriers.data(), 0, nullptr);
        }

        commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,    // Bind point
                pipelineLayout,             // Pipeline Layout
                0,                                  // First descriptor set
                { descriptorSets[frame] },          // List of descriptor sets
                {});                                // Dynamic offsets

        {
            std::vector<vk::BufferMemoryBarrier> barriers = getMemoryBarriers(
                    vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                        // source and destination queues
                        computeQindex, computeQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        ),
                    frame
                    );

            auto pipelineIt = pipelines.begin();
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineIt);
            commandBuffer.dispatch(groups_x, groups_y, groups_z);
            // Need barriers between passes
            for (; pipelineIt != pipelines.end(); ++pipelineIt) {
                commandBuffer.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        {}, // dependency flags
                        0, nullptr, barriers.size(), barriers.data(), 0, nullptr);
            }
        }

        if (graphicsQindex != computeQindex) {
            // Make sure that graphics pipeline doesn't read incomplete data
            std::vector<vk::BufferMemoryBarrier> barriers = getMemoryBarriers(
                    vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        vk::AccessFlagBits::eShaderWrite, {},
                        // source and destination queues
                        computeQindex, graphicsQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        ),
                    frame
                    );
            commandBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eDrawIndirect,
                    {}, // dependency flags
                    0, nullptr, barriers.size(), barriers.data(), 0, nullptr);
        }

        commandBuffer.end();

        // Create the corresponding barriers for graphics pipeline
        graphicsAcquireBarriers[frame] = getMemoryBarriers(
                vk::BufferMemoryBarrier(
                    // source mask, destination mask
                    {}, vk::AccessFlagBits::eVertexAttributeRead,
                    // source and destination queues
                    computeQindex, graphicsQindex,
                    // buffer range
                    VK_NULL_HANDLE, 0, 0
                    ),
                frame
                );
        graphicsReleaseBarriers[frame] = getMemoryBarriers(
                vk::BufferMemoryBarrier(
                    // source mask, destination mask
                    vk::AccessFlagBits::eVertexAttributeRead, {},
                    // source and destination queues
                    graphicsQindex, computeQindex,
                    // buffer range
                    VK_NULL_HANDLE, 0, 0
                    ),
                frame
                );
    }

    // Immediately release the buffers for initial synchronization
    // Instead of doing this, you could also skip the first acquire in the graphics pipeline
    // This trades initialization speed for render speed (no need to check if first acquire)
    if (graphicsQindex != computeQindex) {
        vk::CommandBuffer oneShot = context->device->allocateCommandBuffers(
                vk::CommandBufferAllocateInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1)
                )[0];

        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        oneShot.begin(beginInfo);

        uint32_t totalReleases = storageBuffers.size() + uniformBuffers.size() * MAX_FRAMES_IN_FLIGHT;
        std::vector<vk::BufferMemoryBarrier> releaseBarrier(
                totalReleases,
                vk::BufferMemoryBarrier(
                    // source mask, destination mask
                    vk::AccessFlagBits::eShaderWrite, {},
                    // source and destination queues
                    computeQindex, graphicsQindex,
                    // buffer range
                    VK_NULL_HANDLE, 0, 0
                    )
                );

        uint32_t i = 0;
        for (Descriptor* descriptor : descriptors) {
            if (descriptor->type == Descriptor::eSSBO) {
                auto obj = ((ComputeStorage*)descriptor);
                releaseBarrier[i].buffer = obj->buffer;
                releaseBarrier[i].size = obj->size;
            }
            else if (descriptors[i]->type == Descriptor::eFrameUniform) {
                auto obj = ((FrameUniform*)descriptor);
                for (uint32_t j = 0; j < MAX_FRAMES_IN_FLIGHT; ++j) {
                    releaseBarrier[i].buffer = obj->buffers[j];
                    releaseBarrier[i].size = obj->size;
                    ++i;
                }
            }
            ++i;
        }

        oneShot.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eDrawIndirect,
                {}, // dependency flags
                0, nullptr, releaseBarrier.size(), releaseBarrier.data(), 0, nullptr);

        oneShot.end();

        vk::SubmitInfo submitInfo = {};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &oneShot;

        context->computeQueue.submit(submitInfo, nullptr);
        context->computeQueue.waitIdle();

        context->device->freeCommandBuffers(commandPool, oneShot);
    }
}


void ComputeSystem::signalSemaphore() {
    // Signal the semaphore (kickstart)
    vk::SubmitInfo signalSubmit(
            0, nullptr, nullptr, // wait nothing
            0, nullptr, // do nothing
            1, &sem); // just signal
    context->computeQueue.submit(signalSubmit);
}


ComputeSystem::~ComputeSystem() {
    context->device->destroySemaphore(sem);

    // Compute pipeline cleanup
    for (auto pipeline : pipelines) context->device->destroyPipeline(pipeline);
    context->device->destroyPipelineLayout(pipelineLayout);
    context->device->destroyDescriptorPool(descriptorPool);
    context->device->destroyDescriptorSetLayout(descriptorSetLayout);

    context->device->destroyCommandPool(commandPool);

    for (auto storage : storageBuffers) {
        context->device->destroyBuffer(storage.buffer);
        context->device->freeMemory(storage.memory);
    }
}
