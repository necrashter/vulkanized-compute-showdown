#include "Texture.h"

#include "stb_image.h"
#include <math.h>

#ifdef USE_LIBKTX
#include <ktx.h>
#endif

namespace {
    constexpr vk::SamplerCreateInfo defaultSamplerInfo({},
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
            false, 0.0f,
            // comparison function
            false,
            vk::CompareOp::eAlways,
            // min lod & max lod
            // 0.5f * (float)mipLevels, (float)mipLevels,  // use this to force mipmap
            // 0.0f, (float)mipLevels,
            0.0f, 1.0f,
            // border color
            vk::BorderColor::eIntOpaqueBlack,
            // unnormalized coordinates
            false
            );
}


void TextureImage::load(VulkanContext* context, const char* filename) {
    // Read image
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("Failed to load texture image");
    }

    loadFromBuffer(context, pixels, imageSize, texWidth, texHeight);

    // No need for pixels now
    stbi_image_free(pixels);
}


void TextureImage::loadFromBuffer(
                      VulkanContext* context,
                      void* buffer,
                      VkDeviceSize bufferSize,
                      int texWidth,
                      int texHeight,
                      vk::Format imageFormat) {
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
    auto samplerInfo = defaultSamplerInfo;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = deviceProperties.limits.maxSamplerAnisotropy;
    sampler = context->device->createSampler(samplerInfo);
    // might add try catch
}


void TextureImage::cleanup(VulkanContext* context) {
    context->device->destroySampler(sampler);
    context->device->destroyImageView(view);
    context->device->destroyImage(image);
    context->device->freeMemory(memory);
}


#ifdef USE_LIBKTX
void TextureKTX::load(VulkanContext* context, const char* filename) {
    ktxTexture* kTexture;
    KTX_error_code ktxresult;

    ktxresult = ktxTexture_CreateFromNamedFile(
            filename,
            KTX_TEXTURE_CREATE_NO_FLAGS,
            &kTexture);

    if (KTX_SUCCESS != ktxresult) {
        std::stringstream message;
        message << "Reading KTX texture \"" << filename << "\" failed: " << ktxErrorString(ktxresult);
        throw std::runtime_error(message.str());
    }

    ktxresult = ktxTexture_VkUploadEx(kTexture, &context->ktxInfo, &texture,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (KTX_SUCCESS != ktxresult) {
        std::stringstream message;
        message << "KTX texture upload failed: " << ktxErrorString(ktxresult);
        throw std::runtime_error(message.str());
    }

    ktxTexture_Destroy(kTexture);

    vk::PhysicalDeviceProperties deviceProperties = context->physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo = defaultSamplerInfo;
    samplerInfo.maxLod = texture.levelCount;
    samplerInfo.anisotropyEnable = true;
    samplerInfo.maxAnisotropy = deviceProperties.limits.maxSamplerAnisotropy;
    sampler = context->device->createSampler(samplerInfo);

    vk::ImageViewCreateInfo viewInfo;
    viewInfo.image = texture.image;
    viewInfo.format = static_cast<vk::Format>(texture.imageFormat);
    viewInfo.viewType = static_cast<vk::ImageViewType>(texture.viewType);
    viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    viewInfo.subresourceRange.layerCount = texture.layerCount;
    viewInfo.subresourceRange.levelCount = texture.levelCount;
    view = context->device->createImageView(viewInfo);
}

void TextureKTX::cleanup(VulkanContext* context) {
    context->device->destroyImageView(view);
    context->device->destroySampler(sampler);
    ktxVulkanTexture_Destruct(&texture, context->device.get(), nullptr);
}
#endif
