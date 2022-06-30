#pragma once

#include "VulkanContext.h"
#include <vulkan/vulkan.hpp>


class TextureImage {
public:
    vk::Image image;
    vk::DeviceMemory memory;
    vk::ImageView view;
    vk::Sampler sampler;
    uint32_t mipLevels;

    // Load from a regular image file with stb
    void load(VulkanContext* context, const char* filename);

    // Load from a given buffer
    void loadFromBuffer(
            VulkanContext* context,
            void* buffer,
            VkDeviceSize bufferSize,
            int texWidth,
            int texHeight,
            vk::Format imageFormat = vk::Format::eR8G8B8A8Srgb);

    void cleanup(VulkanContext* context);
};


class Material {
public:
    TextureImage texture;
};


#ifdef USE_LIBKTX
class TextureKTX {
public:
    ktxVulkanTexture texture;
    vk::ImageView view;
    vk::Sampler sampler;

    void load(VulkanContext* context, const char* filename);
    void cleanup(VulkanContext* context);
};
#endif
