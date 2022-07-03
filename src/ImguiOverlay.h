#ifndef IMGUI_OVERLAY_H
#define IMGUI_OVERLAY_H

#include "VulkanContext.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

class ImguiOverlay {
private:
    VulkanContext* const context;
    vk::DescriptorPool descriptorPool;

public:
    ImguiOverlay(VulkanContext* context): context(context) {
    }

    void init(GLFWwindow* window, VkRenderPass renderPass) {
        vk::DescriptorPoolSize poolSizes[] = {
            {vk::DescriptorType::eSampler, 1000},
            {vk::DescriptorType::eCombinedImageSampler, 1000},
            {vk::DescriptorType::eSampledImage, 1000},
            {vk::DescriptorType::eStorageImage, 1000},
            {vk::DescriptorType::eUniformTexelBuffer, 1000},
            {vk::DescriptorType::eStorageTexelBuffer, 1000},
            {vk::DescriptorType::eUniformBuffer, 1000},
            {vk::DescriptorType::eStorageBuffer, 1000},
            {vk::DescriptorType::eUniformBufferDynamic, 1000},
            {vk::DescriptorType::eStorageBufferDynamic, 1000},
            {vk::DescriptorType::eInputAttachment, 1000},
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo(
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                1000, std::size(poolSizes), poolSizes);

        descriptorPool = context->device->createDescriptorPool(descriptorPoolInfo);

        ImGui::CreateContext();

        ImGui_ImplGlfw_InitForVulkan(window, true);

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = context->instance.get();
        init_info.PhysicalDevice = context->physicalDevice;
        init_info.Device = context->device.get();
        init_info.Queue = context->graphicsQueue;
        init_info.DescriptorPool = descriptorPool;
        init_info.MinImageCount = 3;
        init_info.ImageCount = 3;
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

        ImGui_ImplVulkan_Init(&init_info, renderPass);

        auto cmd = context->beginOneShotGraphics();
        ImGui_ImplVulkan_CreateFontsTexture(cmd);
        context->endOneShotGraphics(cmd);

        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }

    void newFrame() {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void render(vk::CommandBuffer cmd) {
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    }

    void cleanup() {
        context->device->destroyDescriptorPool(descriptorPool);
        ImGui_ImplVulkan_Shutdown();
    }
};

inline void ImGuiTooltip(const char* text) {
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 25.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

inline void ImGuiTooltipCombo(const char* name, int* index, const char** names, const char** tooltips, int size) {
    if (ImGui::BeginCombo(name, names[*index])) {
        for (int n = 0; n < size; ++n) {
            const bool is_selected = (*index == n);
            if (ImGui::Selectable(names[n], is_selected)) *index = n;
            if (is_selected) ImGui::SetItemDefaultFocus();
            ImGuiTooltip(tooltips[n]);
        }
        ImGui::EndCombo();
    }
}

#endif
