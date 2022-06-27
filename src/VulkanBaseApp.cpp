#include "VulkanBaseApp.h"
#include "ModelViewScreen.h"
#include "log.h"

bool listGPUs = false;
std::optional<int> selectedGPU = std::nullopt;

const std::vector<std::pair<std::string, std::function<AppScreen*(VulkanBaseApp*)>>> 
screenCreators = {
    {"ModelView", [](VulkanBaseApp* app) { return new ModelViewScreen(app); } },
};

std::function<AppScreen*(VulkanBaseApp*)> findScreen(std::string& query) {
    std::function<AppScreen*(VulkanBaseApp*)> f = nullptr;
    for (auto s : screenCreators) {
        if (s.first == query) {
            f = s.second;
            break;
        }
    }
    return f;
}

/* 
    VULKAN CONTEXT INIT
*/

void VulkanBaseApp::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    auto appInfo = vk::ApplicationInfo(
        ProgramInfo.name,
        VK_MAKE_VERSION(ProgramInfo.version.major, ProgramInfo.version.minor, ProgramInfo.version.patch),
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

void VulkanBaseApp::setupDebugCallback() {
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

void VulkanBaseApp::createSurface() {
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &rawSurface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    surface = rawSurface;
}

void VulkanBaseApp::pickPhysicalDevice() {
    auto devices = instance->enumeratePhysicalDevices();
    if (devices.size() == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }

    if (listGPUs) {
        printGPUs(devices);
    }

    if (selectedGPU) {
        uint32_t index;
        if (*selectedGPU >= 0 && (index = *selectedGPU) < devices.size()) {
            if (isDeviceSuitable(devices[index])) {
                physicalDevice = devices[index];
                goto deviceFound;
            } else {
                std::cerr << "[INIT] Selected GPU (" << *selectedGPU << ") is not suitable!" << std::endl;
            }
        } else {
            std::cerr << "[INIT] Invalid device index: " << *selectedGPU << " (must be between 0 and " << devices.size()-1 << ")" << std::endl;
        }
    }

    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            goto deviceFound;
        }
    }

    throw std::runtime_error("Failed to find a suitable GPU");
deviceFound:
    std::cout << "[INIT] Selected device: " << physicalDevice.getProperties().deviceName << std::endl;
}

void VulkanBaseApp::createLogicalDevice() {
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
    IMGUI
*/

#ifdef USE_IMGUI
void VulkanBaseApp::drawImgui() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Screen")) {
            if (ImGui::MenuItem("Empty")) {
                removeScreen();
            }
            for (auto it : screenCreators) {
                if (ImGui::MenuItem(it.first.c_str())) {
                    removeScreen();
                    screen = it.second(this);
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window")) {
            ImGui::MenuItem("Performance", NULL, &imguiShowPerformance);
            ImGui::MenuItem("About", NULL, &imguiShowAbout);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (imguiShowPerformance) {
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav;
        if (ImGui::Begin("Performance", &imguiShowPerformance, windowFlags)) {
            ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("Average: %d FPS", framesPerSecond);
        }
        ImGui::End();
    }

    if (imguiShowAbout) {
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav;
        if (ImGui::Begin("About", &imguiShowAbout, windowFlags)) {
            ImGui::Text("%s", ProgramInfoStr.c_str());
            if (enableValidationLayers) {
                ImGui::Text(" + Validation layers enabled");
            } else {
                ImGui::Text(" - Validation layers disabled");
            }
        }
        ImGui::End();
    }

    // ImGui::ShowDemoWindow();
}
#endif
