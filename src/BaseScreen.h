#pragma once

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <string>
#include <vector>


// Forward declaration
// VulkanBaseApp depends on this
class VulkanBaseApp;


class BaseScreen {
public:
    VulkanBaseApp* const app;

    BaseScreen(VulkanBaseApp* app): app(app) {}

    // Record render commands that will be submitted to graphics queue
    virtual void recordRenderCommands(vk::RenderPassBeginInfo, vk::CommandBuffer, uint32_t) = 0;

    // Submit the given graphics command buffer in this function.
    // Override to wait for your own semaphores, submit commands to compute shaders, etc.
    // The default implementation submits the commands normally, and can be used as template.
    virtual void submitGraphics(const vk::CommandBuffer* bufferToSubmit, uint32_t currentFrame);

    virtual void mouseMovementCallback(GLFWwindow* window, double xpos, double ypos) = 0;
    virtual void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods) { }

    virtual void update(float delta) = 0;

#ifdef USE_IMGUI
    virtual void imgui() = 0;
#endif

    virtual ~BaseScreen() {}
};


extern const std::vector<std::pair<std::string, std::function<BaseScreen*(VulkanBaseApp*)>>>
screenCreators;

std::function<BaseScreen*(VulkanBaseApp*)> findScreen(std::string& query);
