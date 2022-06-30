#ifndef APPLICATION_H
#define APPLICATION_H

#include "VulkanContext.h"
#include "BaseScreen.h"

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

#ifdef USE_LIBKTX
#include <ktxvulkan.h>
#endif

#include "config.h"
#include "log.h"


const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};


struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> dedicatedComputeFamily;
    std::optional<uint32_t> dedicatedTransferFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};


extern bool listGPUs;
extern bool useDedicatedComputeQueue;
extern std::optional<int> selectedGPU;


class VulkanBaseApp : public VulkanContext {
public:
    GLFWwindow* window;
    float time;
    float delta;
    int framesPerSecond = 0;

    // Currently active screen
    BaseScreen* screen = nullptr;

    size_t currentFrame = 0;
    vk::Extent2D swapChainExtent;

    vk::RenderPass renderPass;

    std::vector<vk::CommandBuffer, std::allocator<vk::CommandBuffer>> commandBuffers;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;

    std::string deviceName;

    // If ImGUI is enabled, this will render the UI to commandBuffer.
    // Otherwise, it will do nothing.
    inline void renderUI(vk::CommandBuffer commandBuffer) {
#ifdef USE_IMGUI
        imguiOverlay.render(commandBuffer);
#endif
    }

private:
#ifdef USE_IMGUI
    ImguiOverlay imguiOverlay;

    bool imguiShowPerformance = false;
    bool imguiShowAbout = false;
    bool imguiErrorPopup = false;
    std::string errorMessage;

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

        window = glfwCreateWindow(WIDTH, HEIGHT, ProgramInfo.name, nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetCursorPosCallback(window, mouseMovementCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<VulkanBaseApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static void mouseMovementCallback(GLFWwindow* window, double xpos, double ypos) {
        auto app = reinterpret_cast<VulkanBaseApp*>(glfwGetWindowUserPointer(window));
        if (app->screen) {
            app->screen->mouseMovementCallback(window, xpos, ypos);
        }
    }

    void initVulkan() {
        createInstance();
        setupDebugCallback();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();

        createCommandPool();

#ifdef USE_LIBKTX
        ktxVulkanDeviceInfo_Construct(
                &ktxInfo,
                physicalDevice,
                device.get(),
                transferQueue,
                transferCommandPool,
                nullptr);
#endif

        // Determine depth format (required before createRenderPass)
        depthFormat = findSupportedFormat(
                {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                vk::ImageTiling::eOptimal,
                vk::FormatFeatureFlagBits::eDepthStencilAttachment
                );

        createSwapChain();

        createRenderPass();

        createImageViews();
        createDepthResources();
        // Create frame buffers (requires depthImageView to be ready)
        createFramebuffers();

        createCommandBuffers();
        createSyncObjects();

#ifdef USE_IMGUI
        imguiOverlay.init(window, renderPass);
#endif
    }

    inline vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
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
        static auto lastTime = startTime;
        static int frames = 0;

        while (!glfwWindowShouldClose(window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
            delta = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
            ++frames;
            if (std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastSecond).count() >= 1.0) {
                lastSecond = std::chrono::high_resolution_clock::now();
                framesPerSecond = frames;
#ifndef USE_IMGUI
                TLOG("Application") << "FPS: " << framesPerSecond << std::endl;
#endif
                frames = 0;
            }

            glfwPollEvents();

#ifdef USE_IMGUI
            imguiOverlay.newFrame();

            drawImgui();
            if (screen) screen->imgui();

            ImGui::Render();
#endif

            if (screen) screen->update(delta);

            renderFrame();
            lastTime = currentTime;
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

    void createInstance();
    void setupDebugCallback();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();

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

        uint32_t indices[] = { queueFamilyIndices.graphics, queueFamilyIndices.present };

        if (queueFamilyIndices.graphics != queueFamilyIndices.present) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = indices;
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
            throw std::runtime_error("Failed to create swap chain!");
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
            throw std::runtime_error("Failed to create render pass!");
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
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        try {
            {
                vk::CommandPoolCreateInfo poolInfo(
                        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                        queueFamilyIndices.graphics
                        );

                commandPool = device->createCommandPool(poolInfo);
            }

            if (queueFamilyIndices.graphics != queueFamilyIndices.transfer) {
                vk::CommandPoolCreateInfo poolInfo(
                        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                        queueFamilyIndices.transfer
                        );

                transferCommandPool = device->createCommandPool(poolInfo);
            } else {
                transferCommandPool = commandPool;
            }
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("Failed to create command pool!");
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
            throw std::runtime_error("Failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(vk::CommandBuffer commandBuffer, size_t imageIndex) {
        vk::CommandBufferBeginInfo beginInfo = {};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

        try {
            commandBuffer.begin(beginInfo);
        }
        catch (vk::SystemError const &err) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }

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

        vk::ClearValue clearValues[] = {
            vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f })),
                vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0)),
        };
        vk::RenderPassBeginInfo renderPassInfo(
                renderPass, 
                swapChainFramebuffers[imageIndex],
                vk::Rect2D({0, 0}, swapChainExtent),
                std::size(clearValues), clearValues
                );

        if (screen) {
            screen->recordRenderCommands(renderPassInfo, commandBuffer, currentFrame);
        } else {
            commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            renderUI(commandBuffer);
            commandBuffer.endRenderPass();
        }

        try {
            commandBuffer.end();
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("Failed to record command buffer!");
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
            throw std::runtime_error("Failed to create synchronization objects for a frame!");
        }
    }

    inline void renderFrame() {
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
            throw std::runtime_error("Failed to acquire swap chain image!");
        }

        (void) device->resetFences(1, &inFlightFences[currentFrame]);

        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        vk::CommandBuffer* bufferToSubmit = &commandBuffers[currentFrame];
        if (screen) {
            screen->submitGraphics(bufferToSubmit, currentFrame);
        } else {
            vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
            vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
            vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };

            vk::SubmitInfo submitInfo(
                    std::size(waitSemaphores), waitSemaphores, waitStages,
                    1, bufferToSubmit,
                    std::size(signalSemaphores), signalSemaphores);

            try {
                graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);
            } catch (vk::SystemError const &err) {
                throw std::runtime_error("Failed to submit draw command buffer");
            }
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
            throw std::runtime_error("Failed to present swap chain image");
        }

        if (resultPresent == vk::Result::eSuboptimalKHR || resultPresent == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
            return;
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }   

    /*
        CHOOOSER FUNCTIONS
    */

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
        /*
         * Mailbox: FPS dips with ImGui+SampleScreen, achieves highest FPS but very inconsistent
         *          tearing impossible
         * Immediate: Consistent high FPS, not as high as mailbox; also tearing possible
         * FIFO: 60 FPS constant, tearing impossible
         * https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPresentModeKHR.html
         */

        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
            if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
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
            if (queueFamily.queueCount <= 0) {
                continue;
            }
            bool graphics = (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) == vk::QueueFlagBits::eGraphics;
            bool compute = (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) == vk::QueueFlagBits::eCompute;
            bool present = device.getSurfaceSupportKHR(i, surface);
            bool transfer = (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) == vk::QueueFlagBits::eTransfer;

            if (graphics) {
                if (!indices.graphicsFamily.has_value()) indices.graphicsFamily = i;
            }
            if (compute) {
                if (!indices.computeFamily.has_value()) indices.computeFamily = i;
            }
            if (compute && !graphics) {
                if (!indices.dedicatedComputeFamily.has_value()) indices.dedicatedComputeFamily = i;
            }
            if (transfer && !graphics) {
                if (!indices.dedicatedTransferFamily.has_value()) indices.dedicatedTransferFamily = i;
            }
            if (present) {
                if (!indices.presentFamily.has_value()) indices.presentFamily = i;
            }

            ++i;
        }

        return indices;
    }

    inline void printGPUs(std::vector<vk::PhysicalDevice> &devices) {
        std::cout << std::endl;
        int deviceIndex = 0;
        for (const auto& device : devices) {
            auto props = device.getProperties();
            std::cout << "Device " << deviceIndex << ": " << props.deviceName
                << "\n\tVulkan Version: " << VK_VERSION_MAJOR(props.apiVersion) << "." << VK_VERSION_MINOR(props.apiVersion) << "." << VK_VERSION_PATCH(props.apiVersion)
                << "\n\tType: ";
            switch (props.deviceType) {
                case vk::PhysicalDeviceType::eDiscreteGpu:
                    std::cout << "Discrete GPU";
                    break;
                case vk::PhysicalDeviceType::eIntegratedGpu:
                    std::cout << "Integrated GPU";
                    break;
                case vk::PhysicalDeviceType::eCpu:
                    std::cout << "CPU";
                    break;
                case vk::PhysicalDeviceType::eVirtualGpu:
                    std::cout << "Virtual GPU";
                    break;
                case vk::PhysicalDeviceType::eOther:
                    std::cout << "Other";
                    break;
            }
            std::cout << "\n\tQueue Families:";
            int i = 0;
            for (const auto& qfam : device.getQueueFamilyProperties()) {
                std::cout << "\n\t\tQ" << i << " (count=" << qfam.queueCount << "):";
                if (qfam.queueFlags & vk::QueueFlagBits::eGraphics) {
                    std::cout << " graphics";
                }
                if (qfam.queueFlags & vk::QueueFlagBits::eCompute) {
                    std::cout << " compute";
                }
                if (qfam.queueFlags & vk::QueueFlagBits::eTransfer) {
                    std::cout << " transfer";
                }
                if (device.getSurfaceSupportKHR(i, surface)) {
                    std::cout << " present";
                }
                ++i;
            }
            std::cout << std::endl;
            ++deviceIndex;
        }
        std::cout << std::endl;
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
    VulkanBaseApp()
#ifdef USE_IMGUI
        : imguiOverlay(this)
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
        try {
            removeScreen();
        } catch(vk::DeviceLostError&) {
            std::cerr << "Device Lost error in VulkanBaseApp cleanup" << std::endl;
            goto glfwCleanup;
        }

        // NOTE: instance destruction is handled by UniqueInstance, same for device

        cleanupSwapChain();

        device->destroyRenderPass(renderPass);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            device->destroySemaphore(renderFinishedSemaphores[i]);
            device->destroySemaphore(imageAvailableSemaphores[i]);
            device->destroyFence(inFlightFences[i]);
        }

#ifdef USE_LIBKTX
        ktxVulkanDeviceInfo_Destruct(&ktxInfo);
#endif

        if (commandPool != transferCommandPool) {
            device->destroyCommandPool(transferCommandPool);
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

glfwCleanup:
        glfwDestroyWindow(window);

        glfwTerminate();
    }
};


#endif
