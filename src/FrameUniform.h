#pragma once

#include "VulkanBaseApp.h"

class FrameUniform {
public:
    VulkanContext* const app;
    std::vector<vk::Buffer> buffers;
    std::vector<vk::DeviceMemory> memories;

    std::vector<void*> mappings;

    FrameUniform(VulkanContext* app, size_t size) : app(app) {
        vk::DeviceSize uniformBufferSize = size;
        buffers.resize(MAX_FRAMES_IN_FLIGHT);
        memories.resize(MAX_FRAMES_IN_FLIGHT);
        mappings.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            app->createBuffer(
                    uniformBufferSize,
                    vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                    buffers[i], memories[i]
                    );
            mappings[i] = app->device->mapMemory(memories[i], 0, size);
        }
    }

    ~FrameUniform() {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            app->device->destroyBuffer(buffers[i]);
            app->device->freeMemory(memories[i]);
        }
    }
};
