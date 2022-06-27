#include "VulkanContext.h"



#ifdef NDEBUG
bool enableValidationLayers = false;
#else
bool enableValidationLayers = true;
#endif


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, callback, pAllocator);
    }
}


bool checkValidationLayerSupport() {
    auto availableLayers = vk::enumerateInstanceLayerProperties();
    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}


/*
   BUFFER
   */

void VulkanContext::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
    vk::BufferCreateInfo bufferInfo = {};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;

    try {
        buffer = device->createBuffer(bufferInfo);
    }
    catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create buffer");
    }

    vk::MemoryRequirements memRequirements = device->getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo = {};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    try {
        bufferMemory = device->allocateMemory(allocInfo);
    }
    catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to allocate buffer memory");
    }

    device->bindBufferMemory(buffer, bufferMemory, 0);
}


uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}


void VulkanContext::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    vk::CommandBuffer commandBuffer = beginOneShotCommands();

    vk::BufferCopy copyRegion = {};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

    endOneShotCommands(commandBuffer);
}


/*
   IMAGE
   */

void VulkanContext::createImage(
        uint32_t width, uint32_t height,
        uint32_t mipLevels,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::Image& image,
        vk::DeviceMemory& memory
        ) {
    vk::ImageCreateInfo createInfo({},
            vk::ImageType::e2D,
            format,
            {width, height, 1},
            mipLevels, // mip levels
            1, // array levels
            vk::SampleCountFlagBits::e1,
            tiling, usage,
            vk::SharingMode::eExclusive);

    image = device->createImage(createInfo);

    vk::MemoryRequirements memreq = device->getImageMemoryRequirements(image);

    vk::MemoryAllocateInfo allocInfo(
            memreq.size,
            findMemoryType(memreq.memoryTypeBits, properties)
            );
    memory = device->allocateMemory(allocInfo);

    device->bindImageMemory(image, memory, 0);
}

void VulkanContext::transitionImageLayout(
        vk::Image image, vk::Format format,
        uint32_t mipLevels,
        vk::ImageLayout oldLayout, vk::ImageLayout newLayout
        ) {
    vk::CommandBuffer commandBuffer = beginOneShotCommands();

    vk::ImageMemoryBarrier barrier(
            {}, {},
            oldLayout, newLayout,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            image,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1));

    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;

        if (hasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
        }
    }

    vk::PipelineStageFlags srcStage, dstStage;

    if (oldLayout == vk::ImageLayout::eUndefined) {
        if (newLayout == vk::ImageLayout::eTransferDstOptimal) {
            // Transfer writes don't need to wait for anything
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
            dstStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

            srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
            dstStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else {
            throw std::invalid_argument("Unsupported layout transition");
        }
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eReadOnlyOptimal) {
        // Shader reading should wait for transfer writes
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        srcStage = vk::PipelineStageFlagBits::eTransfer;
        dstStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
        throw std::invalid_argument("Unsupported layout transition");
    }

    commandBuffer.pipelineBarrier(
            srcStage, dstStage,
            (vk::DependencyFlags) 0,
            0, nullptr,
            0, nullptr,
            1, &barrier
            );

    endOneShotCommands(commandBuffer);
}

void VulkanContext::copyBufferToImage(
        vk::Buffer buffer, vk::Image image,
        uint32_t width, uint32_t height
        ) {
    vk::CommandBuffer commandBuffer = beginOneShotCommands();

    vk::BufferImageCopy region(
            0, 0, 0,
            vk::ImageSubresourceLayers(
                vk::ImageAspectFlagBits::eColor,
                0, 0, 1),
            {0, 0, 0},
            {width, height, 1});

    commandBuffer.copyBufferToImage(
            buffer, image,
            vk::ImageLayout::eTransferDstOptimal,
            1, &region);

    endOneShotCommands(commandBuffer);
}


void VulkanContext::generateMipmaps(vk::Image image, vk::Format format, int32_t w, int32_t h, uint32_t mipLevels) {
    auto formatProperties = physicalDevice.getFormatProperties(format);
    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("Image format doesn't support linear blitting (required for generating mipmaps");
        // Alternative solutions you can implement in this case:
        // software resize e.g. stb_image_resize
        // Load mipmap levels from file (for better loading time)
    }
    vk::CommandBuffer commandBuffer = beginOneShotCommands();

    // This barrier is used to change the type of mip source
    vk::ImageMemoryBarrier mipSourceBarrier(
            // src & dst access flags
            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
            // old & new layouts
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
            // src & dst q family indices
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            image,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

    // this barrier is used to convert a mip level to shader read-only optimal layout
    vk::ImageMemoryBarrier finalBarrier(
            // src & dst access flags
            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
            // old & new layouts
            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            // src & dst q family indices
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            image,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

    for (uint32_t i = 1; i < mipLevels; ++i) {
        int32_t nw = w > 1 ? w/2 : 1;
        int32_t nh = h > 1 ? h/2 : 1;

        mipSourceBarrier.subresourceRange.baseMipLevel = i-1;
        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                (vk::DependencyFlags) 0,
                0, nullptr,
                0, nullptr,
                1, &mipSourceBarrier);

        vk::ImageBlit blit(
                vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i-1, 0, 1),
                { vk::Offset3D{0, 0, 0}, vk::Offset3D{w, h, 1} },
                vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1),
                { vk::Offset3D{0, 0, 0}, vk::Offset3D{nw, nh, 1} }
                );

        // NOTE: In case you create a dedicated transfer queue: blit image neeeds graphicsQueue
        commandBuffer.blitImage(
                image, vk::ImageLayout::eTransferSrcOptimal,
                image, vk::ImageLayout::eTransferDstOptimal,
                1, &blit,
                vk::Filter::eLinear);

        finalBarrier.subresourceRange.baseMipLevel = i-1;
        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
                (vk::DependencyFlags) 0,
                0, nullptr,
                0, nullptr,
                1, &finalBarrier);

        w = nw; h = nh;
    }

    finalBarrier.subresourceRange.baseMipLevel = mipLevels-1;
    // Note that the previous layout of last mip level is different
    finalBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
            (vk::DependencyFlags) 0,
            0, nullptr,
            0, nullptr,
            1, &finalBarrier);

    endOneShotCommands(commandBuffer);
}
