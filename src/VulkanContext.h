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

extern bool enableValidationLayers;


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback);

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator);


bool checkValidationLayerSupport();



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
    vk::Queue computeQueue;

    vk::CommandPool commandPool;

    struct {
        uint32_t graphics, present, compute;
    } queueFamilyIndices;

////////////////////////////////////////////////////////////////////////
//                         UTILITY FUNCTIONS                          //
////////////////////////////////////////////////////////////////////////

    /*
       BUFFER
    */

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory);

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

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
            );

    void transitionImageLayout(
            vk::Image image, vk::Format format,
            uint32_t mipLevels,
            vk::ImageLayout oldLayout, vk::ImageLayout newLayout
            );

    void copyBufferToImage(
            vk::Buffer buffer, vk::Image image,
            uint32_t width, uint32_t height
            );

    // This function is called on swap chain recreation.
    // Therefore, it is inlined for better performance.
    inline vk::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
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
            throw std::runtime_error("Failed to create image views");
        }
    }

    void generateMipmaps(vk::Image image, vk::Format format, int32_t w, int32_t h, uint32_t mipLevels);

    /*
       COMMANDS
    */

    /*
       ONE SHOT COMMANDS
       These can be moved to a special command buffer for better performance at initialization.
    */

    inline vk::CommandBuffer beginOneShotCommands() {
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

    inline void endOneShotCommands(vk::CommandBuffer commandBuffer) {
        commandBuffer.end();

        vk::SubmitInfo submitInfo = {};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();

        device->freeCommandBuffers(commandPool, commandBuffer);
    }

	/*
	   SHADERS
	   */
    
    inline vk::UniqueShaderModule createShaderModule(const std::vector<char>& code) {
        try {
            return device->createShaderModuleUnique({
                vk::ShaderModuleCreateFlags(),
                code.size(), 
                reinterpret_cast<const uint32_t*>(code.data())
            });
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("Failed to create shader module");
        }
    }
};


#endif
