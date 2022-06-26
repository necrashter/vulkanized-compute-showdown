#ifndef APPLICATION_H
#define APPLICATION_H

#include "VulkanContext.h"
#include "Model.h"
#include "Texture.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
// use 0, 1 depth in Vulkan instead of OpenGL's -1 to 1
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <array>
#include <optional>
#include <set>
#include <chrono>

#ifdef USE_IMGUI
#include "ImguiOverlay.h"
#endif


const int WIDTH = 1280;
const int HEIGHT = 720;

const int MAX_FRAMES_IN_FLIGHT = 2;

const glm::vec3 WORLD_UP(0.0f, 1.0f, 0.0f);

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};


#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback);

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator);


struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

struct UniformBufferObject {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 cameraPosition;
};


class VulkanBaseApp;

class AppScreen {
public:
    VulkanBaseApp* const app;

    AppScreen(VulkanBaseApp* app): app(app) {}

    // Record render commands that will be submitted to graphics queue
    virtual void recordRenderCommands(vk::CommandBuffer commandBuffer, uint32_t index) = 0;

    // Called before graphics commands are submitted.
    virtual void preGraphicsSubmit(uint32_t index) = 0;

    virtual ~AppScreen() {}
};


extern const std::vector<std::pair<std::string, std::function<AppScreen*(VulkanBaseApp*)>>>
screenCreators;


class VulkanBaseApp : public VulkanContext {
public:
    GLFWwindow* window;
    float time;
    int framesPerSecond = 0;

    // Currently active screen
    AppScreen* screen = nullptr;

    size_t currentFrame = 0;
    vk::Extent2D swapChainExtent;

    vk::RenderPass renderPass;

    std::vector<vk::CommandBuffer, std::allocator<vk::CommandBuffer>> commandBuffers;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;

private:
#ifdef USE_IMGUI
    ImguiOverlay imguiOverlay;

    bool imguiShowPerformance = false;

    void drawImgui();
#endif

    vk::Queue presentQueue;

    VkDebugUtilsMessengerEXT callback;
    vk::SurfaceKHR surface;

    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;

    vk::Format depthFormat;
    vk::Image depthImage;
    vk::DeviceMemory depthImageMemory;
    vk::ImageView depthImageView;

    bool framebufferResized = false;

protected:
    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<VulkanBaseApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugCallback();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();

        // Determine depth format (required before createRenderPass)
        depthFormat = findSupportedFormat(
                {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                vk::ImageTiling::eOptimal,
                vk::FormatFeatureFlagBits::eDepthStencilAttachment
                );

        createSwapChain();
        createImageViews();
        createRenderPass();
        createCommandPool();

        createDepthResources();
        // Create frame buffers (requires depthImageView to be ready)
        createFramebuffers();

        createCommandBuffers();
        createSyncObjects();

#ifdef USE_IMGUI
        imguiOverlay.init(window, renderPass);
#endif
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
            vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
        for (vk::Format format : candidates) {
            vk::FormatProperties properties = physicalDevice.getFormatProperties(format);
            if (tiling == vk::ImageTiling::eLinear && (properties.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == vk::ImageTiling::eOptimal && (properties.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        throw std::runtime_error("Failed to find supported format");
    }

    void mainLoop() {
        static auto startTime = std::chrono::high_resolution_clock::now();
        static auto lastSecond = startTime;
        static int frames = 0;

        while (!glfwWindowShouldClose(window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
            ++frames;
            if (std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastSecond).count() >= 1.0) {
                lastSecond = std::chrono::high_resolution_clock::now();
                framesPerSecond = frames;
                frames = 0;
            }

            glfwPollEvents();

#ifdef USE_IMGUI
            imguiOverlay.newFrame();

            drawImgui();

            ImGui::Render();
#endif
            renderFrame();
        }
    }

    void cleanupSwapChain() {
        device->destroyImageView(depthImageView);
        device->destroyImage(depthImage);
        device->freeMemory(depthImageMemory);

        for (auto framebuffer : swapChainFramebuffers) {
            device->destroyFramebuffer(framebuffer);
        }

        for (auto imageView : swapChainImageViews) {
            device->destroyImageView(imageView);
        }

        device->destroySwapchainKHR(swapChain);
    }


    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device->waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        auto appInfo = vk::ApplicationInfo(
            "Hello Triangle",
            VK_MAKE_VERSION(1, 0, 0),
            "-",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_0
        );

        auto extensions = getRequiredExtensions();

        auto createInfo = vk::InstanceCreateInfo(
            vk::InstanceCreateFlags(),
            &appInfo,
            0, nullptr, // enabled layers
            static_cast<uint32_t>(extensions.size()), extensions.data() // enabled extensions
        );

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        try {
            instance = vk::createInstanceUnique(createInfo, nullptr);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void setupDebugCallback() {
        if (!enableValidationLayers) return;

        auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT(
            vk::DebugUtilsMessengerCreateFlagsEXT(),
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            debugCallback,
            nullptr
        );

        // NOTE: Vulkan-hpp has methods for this, but they trigger linking errors...
        //instance->createDebugUtilsMessengerEXT(createInfo);
        //instance->createDebugUtilsMessengerEXTUnique(createInfo);

        // NOTE: reinterpret_cast is also used by vulkan.hpp internally for all these structs
        if (CreateDebugUtilsMessengerEXT(*instance, reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&createInfo), nullptr, &callback) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug callback!");
        }
    }

    void createSurface() {
        VkSurfaceKHR rawSurface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &rawSurface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        surface = rawSurface;
    }

    void pickPhysicalDevice() {
        auto devices = instance->enumeratePhysicalDevices();
        if (devices.size() == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (!physicalDevice) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;

        for (uint32_t queueFamily : uniqueQueueFamilies) {
            queueCreateInfos.push_back({
                vk::DeviceQueueCreateFlags(),
                queueFamily,
                1, // queueCount
                &queuePriority
                });
        }

        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.samplerAnisotropy = true;
        auto createInfo = vk::DeviceCreateInfo(
            vk::DeviceCreateFlags(),
            static_cast<uint32_t>(queueCreateInfos.size()),
            queueCreateInfos.data()
        );
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        try {
            device = physicalDevice.createDeviceUnique(createInfo);
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create logical device!");
        }

        graphicsQueue = device->getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device->getQueue(indices.presentFamily.value(), 0);
    }

    /*
       Swap Chain
    */

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo(
            vk::SwapchainCreateFlagsKHR(),
            surface,
            imageCount,
            surfaceFormat.format,
            surfaceFormat.colorSpace,
            extent,
            1, // imageArrayLayers
            vk::ImageUsageFlagBits::eColorAttachment
        );

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = vk::SwapchainKHR(nullptr);

        try {
            swapChain = device->createSwapchainKHR(createInfo);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create swap chain!");
        }

        swapChainImages = device->getSwapchainImagesKHR(swapChain);

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); ++i) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = vk::SampleCountFlagBits::e1;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

        vk::AttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::AttachmentDescription depthAttachment({},
                depthFormat,
                vk::SampleCountFlagBits::e1,
                vk::AttachmentLoadOp::eClear,
                vk::AttachmentStoreOp::eDontCare,
                vk::AttachmentLoadOp::eDontCare,
                vk::AttachmentStoreOp::eDontCare,
                vk::ImageLayout::eUndefined, // initialLayout
                vk::ImageLayout::eDepthStencilAttachmentOptimal // final
                );
        vk::AttachmentReference depthAttachmentRef(
                1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::SubpassDescription subpass = {};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        vk::SubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
        //dependency.srcAccessMask = 0;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
        dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        vk::AttachmentDescription attachments[] = {colorAttachment, depthAttachment};

        vk::RenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.attachmentCount = 2;
        renderPassInfo.pAttachments = attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        try {
            renderPass = device->createRenderPass(renderPassInfo);
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createDepthResources() {
        createImage(
                swapChainExtent.width,
                swapChainExtent.height,
                1,
                depthFormat,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                depthImage,
                depthImageMemory
                );
        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
        
        transitionImageLayout(depthImage, depthFormat, 1, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
            vk::ImageView attachments[] = {
                swapChainImageViews[i],
                depthImageView,
            };

            vk::FramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 2;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            try {
                swapChainFramebuffers[i] = device->createFramebuffer(framebufferInfo);
            } catch (vk::SystemError const &err) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        vk::CommandPoolCreateInfo poolInfo(
                vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                queueFamilyIndices.graphicsFamily.value()
                );

        try {
            commandPool = device->createCommandPool(poolInfo);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(swapChainFramebuffers.size());

        vk::CommandBufferAllocateInfo allocInfo = {};
        allocInfo.commandPool = commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        try {
            commandBuffers = device->allocateCommandBuffers(allocInfo);
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(size_t i) {
        vk::CommandBufferBeginInfo beginInfo = {};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

        try {
            commandBuffers[i].begin(beginInfo);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        vk::RenderPassBeginInfo renderPassInfo(
                renderPass, 
                swapChainFramebuffers[i],
                vk::Rect2D({0, 0}, swapChainExtent)
                );

        std::array<vk::ClearValue, 2> clearValues {
            vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f })),
                vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0)),
        };
        renderPassInfo.clearValueCount = clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();

        vk::CommandBuffer commandBuffer = commandBuffers[i];

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        {
            vk::Viewport viewport(
                    0.0f, 0.0f, 
                    (float)swapChainExtent.width, (float)swapChainExtent.height,
                    0.0f, 1.0f);

            vk::Rect2D scissor(
                    {0, 0}, // Offset
                    swapChainExtent // extent
                    );

            commandBuffer.setViewport(0, 1, &viewport);
            commandBuffer.setScissor(0, 1, &scissor);

            if (screen) {
                screen->recordRenderCommands(commandBuffer, currentFrame);
            }

#ifdef USE_IMGUI
            imguiOverlay.render(commandBuffer);
#endif
        }
        commandBuffer.endRenderPass();

        try {
            commandBuffer.end();
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        try {
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                imageAvailableSemaphores[i] = device->createSemaphore({});
                renderFinishedSemaphores[i] = device->createSemaphore({});
                inFlightFences[i] = device->createFence({vk::FenceCreateFlagBits::eSignaled});
            }
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }

    void renderFrame() {
        (void) device->waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t imageIndex;
        try {
            vk::ResultValue result = device->acquireNextImageKHR(swapChain, std::numeric_limits<uint64_t>::max(),
                imageAvailableSemaphores[currentFrame], nullptr);
            imageIndex = result.value;
        } catch (vk::OutOfDateKHRError const &err) {
            recreateSwapChain();
            return;
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        (void) device->resetFences(1, &inFlightFences[currentFrame]);

        recordCommandBuffer(imageIndex);

        if (screen) {
            screen->preGraphicsSubmit(currentFrame);
        }

        vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };

        vk::SubmitInfo submitInfo(
                std::size(waitSemaphores), waitSemaphores, waitStages,
                1, &commandBuffers[imageIndex],
                std::size(signalSemaphores), signalSemaphores);

        try {
            graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        vk::PresentInfoKHR presentInfo(
                1, &renderFinishedSemaphores[currentFrame], // wait sem
                1, &swapChain, &imageIndex);

        vk::Result resultPresent;
        try {
            resultPresent = presentQueue.presentKHR(presentInfo);
        } catch (vk::OutOfDateKHRError const &err) {
            resultPresent = vk::Result::eErrorOutOfDateKHR;
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        if (resultPresent == vk::Result::eSuboptimalKHR || resultPresent == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
            return;
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }   

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
            return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
        }

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes) {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
            else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = availablePresentMode;
            }
        }

        return bestMode;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            vk::Extent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device) {
        SwapChainSupportDetails details;
        details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
        details.formats = device.getSurfaceFormatsKHR(surface);
        details.presentModes = device.getSurfacePresentModesKHR(surface);

        return details;
    }

    bool isDeviceSuitable(const vk::PhysicalDevice& device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) {
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : device.enumerateDeviceExtensionProperties()) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
        QueueFamilyIndices indices;

        auto queueFamilies = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            if (queueFamily.queueCount > 0 && device.getSurfaceSupportKHR(i, surface)) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "[VALIDATION] " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

public:
    VulkanBaseApp():
#ifdef USE_IMGUI
        imguiOverlay(this)
#endif
    {
        initWindow();
        initVulkan();
    }

    void run() {
        mainLoop();
    }

    void removeScreen() {
        device->waitIdle();
        if (screen) {
            delete screen;
            screen = nullptr;
        }
    }

    virtual ~VulkanBaseApp() {
        // cleanup();

        // Called by removeScreen
        // device->waitIdle();
        removeScreen();

        // NOTE: instance destruction is handled by UniqueInstance, same for device

        cleanupSwapChain();

        device->destroyRenderPass(renderPass);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            device->destroySemaphore(renderFinishedSemaphores[i]);
            device->destroySemaphore(imageAvailableSemaphores[i]);
            device->destroyFence(inFlightFences[i]);
        }

        device->destroyCommandPool(commandPool);

#ifdef USE_IMGUI
        imguiOverlay.cleanup();
#endif

        // surface is created by glfw, therefore not using a Unique handle
        instance->destroySurfaceKHR(surface);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(*instance, callback, nullptr);
        }

        glfwDestroyWindow(window);

        glfwTerminate();
	}
};



class SampleScreen : public AppScreen {
private:
    Model model;

    struct {
        vk::DescriptorSetLayout perFrame;
        vk::DescriptorSetLayout perMaterial;
    } descriptorSetLayouts;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;

public:
    SampleScreen(VulkanBaseApp* app):
        AppScreen(app),
        model(app)
    {
        model.loadFile("../assets/FlightHelmet/FlightHelmet.gltf");
        model.createBuffers();
        createBuffers();

        createDescriptorPool();
        createDescriptorSetLayout();
        createDescriptorSets();
        createGraphicsPipeline();
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readBinaryFile("shaders/shader.vert.spv");
        auto fragShaderCode = readBinaryFile("shaders/shader.frag.spv");

        auto vertShaderModule = app->createShaderModule(vertShaderCode);
        auto fragShaderModule = app->createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo shaderStages[] = { 
            {
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eVertex,
                *vertShaderModule,
                "main"
            }, 
            {
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eFragment,
                *fragShaderModule,
                "main"
            } 
        };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        auto bindingDescription = Model::vertexBindingDescription;
        auto attributeDescriptions = Model::vertexAttributeDescription;

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        vk::PipelineViewportStateCreateInfo viewportState(
                {},
                1, nullptr,
                1, nullptr // viewport and scissors are dynamic, hence nullptr (ignored)
                );

        vk::PipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizer.depthBiasEnable = VK_FALSE;

        vk::PipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = VK_FALSE;

        vk::PipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = vk::LogicOp::eCopy;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::array<vk::DescriptorSetLayout, 2> setLayouts = {
            descriptorSetLayouts.perFrame,
            descriptorSetLayouts.perMaterial,
        };
        // Push constants information
        std::array<vk::PushConstantRange, 1> pushConstants = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4)),
        };
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({},
                setLayouts.size(), setLayouts.data(),
                pushConstants.size(), pushConstants.data()
                );

        try {
            pipelineLayout = app->device->createPipelineLayout(pipelineLayoutInfo);
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // Depth testing
        vk::PipelineDepthStencilStateCreateInfo depthStencil({}, true, true, vk::CompareOp::eLess, false, false);

        vk::GraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = app->renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = nullptr;
        pipelineInfo.pDepthStencilState = &depthStencil;

        std::array<vk::DynamicState, 2> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates.size(), dynamicStates.data());
        pipelineInfo.pDynamicState = &dynamicState;

        try {
            graphicsPipeline = app->device->createGraphicsPipeline(nullptr, pipelineInfo).value;
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }   

    /*
       Descriptor
   */

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboBinding(
                0, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                nullptr);

        try {
            descriptorSetLayouts.perFrame = app->device->createDescriptorSetLayout({{}, 1, &uboBinding});
            descriptorSetLayouts.perMaterial = model.createMaterialDescriptorSetLayout();
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createBuffers() {
        // Uniform buffers
        vk::DeviceSize uniformBufferSize = sizeof(UniformBufferObject);
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            app->createBuffer(
                    uniformBufferSize,
                    vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                    uniformBuffers[i], uniformBuffersMemory[i]
                    );
        }
    }

    void createDescriptorPool() {
        size_t materialCount = model.materials.size();
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {
            // UBO
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
            // Sampler
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, materialCount),
        };
        vk::DescriptorPoolCreateInfo poolCreateInfo({},
                MAX_FRAMES_IN_FLIGHT + materialCount,
                poolSizes.size(), poolSizes.data());
        try {
            descriptorPool = app->device->createDescriptorPool(poolCreateInfo);
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        try {
            std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayouts.perFrame);
            descriptorSets = app->device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
                        descriptorPool,
                        static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
                        layouts.data()));
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));

            std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {
                // UBO
                vk::WriteDescriptorSet(
                        descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer,
                        nullptr, // image info
                        &bufferInfo
                        ),
            };
            app->device->updateDescriptorSets(
                    descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }

        model.createMaterialDescriptorSets(descriptorPool, descriptorSetLayouts.perMaterial);
    }


    void updateUniformBuffer(uint32_t index) {
        glm::vec3 cameraPosition = glm::vec3(
            glm::rotate(glm::mat4(1.0f), app->time * glm::radians(90.0f), WORLD_UP) * glm::vec4(3.0f, 0.0f, 0.0f, 1.0f)
                );

        UniformBufferObject ubo {
            glm::lookAt(cameraPosition, glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP),
            glm::perspective(glm::radians(60.0f), app->swapChainExtent.width / (float) app->swapChainExtent.height, 0.1f, 10.0f),
            cameraPosition
        };
        // Y coordinate is inverted
        ubo.proj[1][1] *= -1;

        void* data = app->device->mapMemory(uniformBuffersMemory[index], 0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        app->device->unmapMemory(uniformBuffersMemory[index]);
    }

    virtual void preGraphicsSubmit(uint32_t index) override {
        updateUniformBuffer(index);
    }


    virtual void recordRenderCommands(vk::CommandBuffer commandBuffer, uint32_t index) override {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[index], 0, nullptr);

        model.render(commandBuffer, pipelineLayout,
                glm::scale(glm::mat4(1.0f), glm::vec3(3.0f))
                );
    }

    virtual ~SampleScreen() {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            app->device->destroyBuffer(uniformBuffers[i]);
            app->device->freeMemory(uniformBuffersMemory[i]);
        }

        app->device->destroyPipeline(graphicsPipeline);
        app->device->destroyPipelineLayout(pipelineLayout);

        // DescriptorSets are removed automatically with descriptorPool
        app->device->destroyDescriptorPool(descriptorPool);
        app->device->destroyDescriptorSetLayout(descriptorSetLayouts.perFrame);
        app->device->destroyDescriptorSetLayout(descriptorSetLayouts.perMaterial);

        model.cleanup();
    }
};

#endif
