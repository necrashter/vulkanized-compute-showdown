#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#include <vulkan/vulkan.hpp>

#include <fstream>


inline bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}


const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};


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


std::vector<char> readBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}



// Abstract vulkan context, base class for application
class VulkanContext {
public:
    vk::UniqueInstance instance;

    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;

    vk::Queue graphicsQueue;

    vk::CommandPool commandPool;

////////////////////////////////////////////////////////////////////////
//                         UTILITY FUNCTIONS                          //
////////////////////////////////////////////////////////////////////////

    /*
       BUFFER
    */

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
        vk::BufferCreateInfo bufferInfo = {};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;

        try {
            buffer = device->createBuffer(bufferInfo);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create buffer!");
        }

        vk::MemoryRequirements memRequirements = device->getBufferMemoryRequirements(buffer);

        vk::MemoryAllocateInfo allocInfo = {};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        try {
            bufferMemory = device->allocateMemory(allocInfo);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        device->bindBufferMemory(buffer, bufferMemory, 0);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
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

    void createImage(
            uint32_t width, uint32_t height,
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
                1, // mip levels
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

    void transitionImageLayout(
            vk::Image image, vk::Format format,
            vk::ImageLayout oldLayout, vk::ImageLayout newLayout
            ) {
        vk::CommandBuffer commandBuffer = beginOneShotCommands();

        vk::ImageMemoryBarrier barrier(
                {}, {},
                oldLayout, newLayout,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                image,
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

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

    void copyBufferToImage(
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

    vk::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
        vk::ImageViewCreateInfo createInfo = {};
        createInfo.image = image;
        createInfo.viewType = vk::ImageViewType::e2D;
        createInfo.format = format;
        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;
        createInfo.subresourceRange.aspectMask = aspectFlags;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        try {
            return device->createImageView(createInfo);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    /*
       COMMANDS
    */

    /*
       ONE SHOT COMMANDS
       These can be moved to a special command buffer for better performance at initialization.
    */

    vk::CommandBuffer beginOneShotCommands() {
        vk::CommandBufferAllocateInfo allocInfo = {};
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        vk::CommandBuffer commandBuffer = device->allocateCommandBuffers(allocInfo)[0];

        vk::CommandBufferBeginInfo beginInfo = {};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endOneShotCommands(vk::CommandBuffer commandBuffer) {
        commandBuffer.end();

        vk::SubmitInfo submitInfo = {};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();

        device->freeCommandBuffers(commandPool, commandBuffer);
    }
};

#endif