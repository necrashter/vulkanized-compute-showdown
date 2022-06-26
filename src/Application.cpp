#include "Application.h"

const std::vector<std::pair<std::string, std::function<AppScreen*(VulkanBaseApp*)>>> 
screenCreators = {
    {"SampleScreen", [](VulkanBaseApp* app) { return new SampleScreen(app); } },
};


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

#ifdef USE_IMGUI
void VulkanBaseApp::drawImgui() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Scene")) {
            if (ImGui::MenuItem("Remove Screen")) {
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
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (imguiShowPerformance) {
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
        if (ImGui::Begin("Performance", &imguiShowPerformance, windowFlags)) {
            ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("Average: %d FPS", framesPerSecond);
        }
        ImGui::End();
    }

    // ImGui::ShowDemoWindow();
}
#endif
