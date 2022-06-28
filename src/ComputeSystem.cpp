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
    bufferInfos.emplace_back(storage.buffer, 0, storage.size);
    descriptorTypes.emplace_back(vk::DescriptorType::eStorageBuffer);

    return &storage.buffer;
}


void* ComputeSystem::createUniformBuffer(size_t size) {
    if (size == 0) {
        throw std::runtime_error("Compute Shader Uniform Buffer size is 0");
    }
    uniformBuffers.emplace_back();
    auto& storage = uniformBuffers.back();

    context->createBuffer(
            size,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            storage.buffer, storage.memory
            );

    storage.size = size;
    storage.mapping = context->device->mapMemory(storage.memory, 0, size);

    uint32_t bindingIndex = (uint32_t) bindings.size();
    bindings.emplace_back(
            bindingIndex,
            vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eCompute,
            nullptr
            );
    bufferInfos.emplace_back(storage.buffer, 0, storage.size);
    descriptorTypes.emplace_back(vk::DescriptorType::eUniformBuffer);

    return storage.mapping;
}


////////////////////////////////////////////////////////////////////////
//                          COMPUTE PIPELINE                          //
////////////////////////////////////////////////////////////////////////

void ComputeSystem::finalizeLayout() {
    // Create Descriptor Pool
    // ---------------------------------------------------------------

    vk::DescriptorPoolSize poolSizes[] = {
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, uniformBuffers.size()),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, storageBuffers.size()),
    };
    vk::DescriptorPoolCreateInfo poolCreateInfo({},
            1,
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
        vk::DescriptorSetLayout layouts[] = { descriptorSetLayout };
        descriptorSet = context->device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
                    descriptorPool,
                    std::size(layouts), layouts))[0];
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    {
        std::vector<vk::WriteDescriptorSet> descriptorWrites(
                bufferInfos.size(), 
                vk::WriteDescriptorSet(
                        descriptorSet, 0, 0, 1, {},
                        nullptr, nullptr
                        )
                );
        for (uint32_t i = 0; i < bufferInfos.size(); ++i) {
            descriptorWrites[i].descriptorType = descriptorTypes[i];
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
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


void ComputeSystem::recordCommands(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    // Create Command Pool & Buffer
    {
        vk::CommandPoolCreateInfo poolInfo({}, context->queueFamilyIndices.compute);
        commandPool = context->device->createCommandPool(poolInfo);

        commandBuffer = context->device->allocateCommandBuffers(
                vk::CommandBufferAllocateInfo(
                    commandPool,
                    vk::CommandBufferLevel::ePrimary,
                    1
                    ))[0];
    }

    uint32_t graphicsQindex = context->queueFamilyIndices.graphics;
    uint32_t computeQindex = context->queueFamilyIndices.compute;

    // for some reason we need simultaneous use
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
    if (commandBuffer.begin(&beginInfo) != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to begin recording to compute command buffer");
    }

    if (graphicsQindex != computeQindex) {
        // need memory barrier if graphics and compute queues are different
        // Make sure that graphics pipeline red the previous inputs
        // Matching transfer acquire & release barriers must be present in the graphics pipeline
        std::vector<vk::BufferMemoryBarrier> barriers(
                bufferInfos.size(),
                vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        {}, vk::AccessFlagBits::eShaderWrite,
                        // source and destination queues
                        graphicsQindex, computeQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        )
                );
        for (uint32_t i = 0; i < bufferInfos.size(); ++i) {
            barriers[i].buffer = bufferInfos[i].buffer;
            barriers[i].size = bufferInfos[i].range;
        }
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
            { descriptorSet },          // List of descriptor sets
            {});                                // Dynamic offsets

    {
        std::vector<vk::BufferMemoryBarrier> barriers(
                bufferInfos.size(),
                vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                        // source and destination queues
                        computeQindex, computeQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        )
                );
        for (uint32_t i = 0; i < bufferInfos.size(); ++i) {
            barriers[i].buffer = bufferInfos[i].buffer;
            barriers[i].size = bufferInfos[i].range;
        }

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
        std::vector<vk::BufferMemoryBarrier> barriers(
                bufferInfos.size(),
                vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        vk::AccessFlagBits::eShaderWrite, {},
                        // source and destination queues
                        computeQindex, graphicsQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        )
                );
        for (uint32_t i = 0; i < bufferInfos.size(); ++i) {
            barriers[i].buffer = bufferInfos[i].buffer;
            barriers[i].size = bufferInfos[i].range;
        }
        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eDrawIndirect,
                {}, // dependency flags
                0, nullptr, barriers.size(), barriers.data(), 0, nullptr);
    }

    commandBuffer.end();

    // Create the corresponding barriers for graphics pipeline
    {
        graphicsAcquireBarriers.resize(
                bufferInfos.size(),
                vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        {}, vk::AccessFlagBits::eVertexAttributeRead,
                        // source and destination queues
                        computeQindex, graphicsQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        )
                );
        graphicsReleaseBarriers.resize(
                bufferInfos.size(),
                vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        vk::AccessFlagBits::eVertexAttributeRead, {},
                        // source and destination queues
                        graphicsQindex, computeQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        )
                );
        for (uint32_t i = 0; i < bufferInfos.size(); ++i) {
            graphicsAcquireBarriers[i].buffer = bufferInfos[i].buffer;
            graphicsAcquireBarriers[i].size = bufferInfos[i].range;
            graphicsReleaseBarriers[i].buffer = bufferInfos[i].buffer;
            graphicsReleaseBarriers[i].size = bufferInfos[i].range;
        }
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

        // std::vector<vk::BufferMemoryBarrier> acquireBarrier(
        //         bufferInfos.size(),
        //         vk::BufferMemoryBarrier(
        //                 // source mask, destination mask
        //                 {}, vk::AccessFlagBits::eShaderWrite,
        //                 // source and destination queues
        //                 graphicsQindex, computeQindex,
        //                 // buffer range
        //                 VK_NULL_HANDLE, 0, 0
        //                 )
        //         );
        // for (uint32_t i = 0; i < bufferInfos.size(); ++i) {
        //     acquireBarrier[i].buffer = bufferInfos[i].buffer;
        //     acquireBarrier[i].size = bufferInfos[i].range;
        // }
        // oneShot.pipelineBarrier(
        //         vk::PipelineStageFlagBits::eDrawIndirect,
        //         vk::PipelineStageFlagBits::eComputeShader,
        //         {}, // dependency flags
        //         0, nullptr, acquireBarrier.size(), acquireBarrier.data(), 0, nullptr);

        std::vector<vk::BufferMemoryBarrier> releaseBarrier(
                bufferInfos.size(),
                vk::BufferMemoryBarrier(
                        // source mask, destination mask
                        vk::AccessFlagBits::eShaderWrite, {},
                        // source and destination queues
                        computeQindex, graphicsQindex,
                        // buffer range
                        VK_NULL_HANDLE, 0, 0
                        )
                );
        for (uint32_t i = 0; i < bufferInfos.size(); ++i) {
            releaseBarrier[i].buffer = bufferInfos[i].buffer;
            releaseBarrier[i].size = bufferInfos[i].range;
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

    for (auto storage : uniformBuffers) {
        context->device->unmapMemory(storage.memory);
        context->device->destroyBuffer(storage.buffer);
        context->device->freeMemory(storage.memory);
    }

}
