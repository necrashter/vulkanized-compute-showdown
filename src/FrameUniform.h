#pragma once

#include "VulkanBaseApp.h"

class Descriptor {
public:
    enum Type {
        eFrameUniform,
        eSSBO,
    };

    Type type;
    Descriptor(Type t): type(t) {}
};

class FrameUniform : public Descriptor {
public:
    VulkanContext* const context;
    size_t size;
    std::vector<vk::Buffer> buffers;
    std::vector<vk::DeviceMemory> memories;

    std::vector<void*> mappings;

    FrameUniform(VulkanContext* context, size_t size) : 
        Descriptor(eFrameUniform),
        context(context), size(size) {
        vk::DeviceSize uniformBufferSize = size;
        buffers.resize(MAX_FRAMES_IN_FLIGHT);
        memories.resize(MAX_FRAMES_IN_FLIGHT);
        mappings.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            context->createBuffer(
                    uniformBufferSize,
                    vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                    buffers[i], memories[i]
                    );
            mappings[i] = context->device->mapMemory(memories[i], 0, size);
        }
    }

    ~FrameUniform() {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            context->device->destroyBuffer(buffers[i]);
            context->device->freeMemory(memories[i]);
        }
    }
};
