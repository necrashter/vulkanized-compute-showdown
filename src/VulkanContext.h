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


inline bool checkValidationLayerSupport() {
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


inline std::vector<char> readBinaryFile(const std::string& filename) {
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

    void transitionImageLayout(
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

    vk::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
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
        createInfo.subresourceRange.levelCount = mipLevels;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        try {
            return device->createImageView(createInfo);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    void generateMipmaps(vk::Image image, vk::Format format, int32_t w, int32_t h, uint32_t mipLevels) {
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
    
    vk::UniqueShaderModule createShaderModule(const std::vector<char>& code) {
        try {
            return device->createShaderModuleUnique({
                vk::ShaderModuleCreateFlags(),
                code.size(), 
                reinterpret_cast<const uint32_t*>(code.data())
            });
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create shader module!");
        }
    }
};

#endif
