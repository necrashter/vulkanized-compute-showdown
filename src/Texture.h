#ifndef TEXTURE_H
#define TEXTURE_H

#include "VulkanContext.h"
#include <vulkan/vulkan.hpp>

#include "stb_image.h"

#include <math.h>


class TextureImage {
public:
    vk::Image image;
    vk::DeviceMemory memory;
    vk::ImageView view;
    vk::Sampler sampler;
	uint32_t mipLevels;

    void load(VulkanContext* context, const char* filename) {
        // Read image
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        loadFromBuffer(context, pixels, imageSize, texWidth, texHeight);

        // No need for pixels now
        stbi_image_free(pixels);
    }

    void loadFromBuffer(
            VulkanContext* context,
            void* buffer,
            VkDeviceSize bufferSize,
            int texWidth,
            int texHeight,
            vk::Format imageFormat = vk::Format::eR8G8B8A8Srgb) {
        // Staging buffer
        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;

        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

        context->createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                stagingBuffer, stagingBufferMemory
                );

        void *data = context->device->mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, buffer, bufferSize);
        context->device->unmapMemory(stagingBufferMemory);

        context->createImage(
                texWidth, texHeight,
                mipLevels,
                imageFormat,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled
                | vk::ImageUsageFlagBits::eTransferSrc, // for generating mipmaps
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                image, memory);
        // initial image layout is eUndefined
        context->transitionImageLayout(
                image, imageFormat,
                mipLevels,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        context->copyBufferToImage(stagingBuffer, image, texWidth, texHeight);

        // To sample the image, one last image layout transition
        // context->transitionImageLayout(
        //         image, imageFormat,
        //         mipLevels,
        //         vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        // NOTE: The operation above will be performed while generating mipmaps

        // Clean up staging buffer
        context->device->destroyBuffer(stagingBuffer);
        context->device->freeMemory(stagingBufferMemory);

        // Generate mipmaps
        context->generateMipmaps(image, imageFormat, texWidth, texHeight, mipLevels);

        // Image view
        view = context->createImageView(image, imageFormat,
                vk::ImageAspectFlagBits::eColor, mipLevels);

        // sampler
        vk::PhysicalDeviceProperties deviceProperties = context->physicalDevice.getProperties();
        vk::SamplerCreateInfo samplerInfo({},
                // mag min filter
                vk::Filter::eLinear, vk::Filter::eLinear,
                // mipmap
                vk::SamplerMipmapMode::eLinear,
                // address mode; uvw
                vk::SamplerAddressMode::eRepeat,
                vk::SamplerAddressMode::eRepeat,
                vk::SamplerAddressMode::eRepeat,
                // mip lod bias
                0.0f,
                // antisotropy
                true,
                deviceProperties.limits.maxSamplerAnisotropy,
                // comparison function
                false,
                vk::CompareOp::eAlways,
                // min lod & max lod
                // 0.5f * (float)mipLevels, (float)mipLevels,  // use this to force mipmap
                0.0f, (float)mipLevels,
                // border color
                vk::BorderColor::eIntOpaqueBlack,
                // unnormalized coordinates
                false
                );
        sampler = context->device->createSampler(samplerInfo);
        // might add try catch
    }

    void cleanup(VulkanContext* context) {
        context->device->destroySampler(sampler);
        context->device->destroyImageView(view);
        context->device->destroyImage(image);
        context->device->freeMemory(memory);
    }
};


class Material {
public:
    TextureImage texture;
};


#endif
