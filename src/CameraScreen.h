#ifndef CAMERA_SCREEN_H
#define CAMERA_SCREEN_H

#include "VulkanBaseApp.h"
#include "Noclip.h"


struct CameraUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 cameraPosition;
};


class CameraScreen : public BaseScreen {
private:
    noclip::cam noclipCam;

protected:
    struct CameraUniform {
        std::vector<vk::Buffer> buffers;
        std::vector<vk::DeviceMemory> memories;
    } cameraUniform;

public:
    CameraScreen(VulkanBaseApp* app):
        BaseScreen(app),
        noclipCam(app->window) {
        vk::DeviceSize uniformBufferSize = sizeof(CameraUBO);
        cameraUniform.buffers.resize(MAX_FRAMES_IN_FLIGHT);
        cameraUniform.memories.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            app->createBuffer(
                    uniformBufferSize,
                    vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                    cameraUniform.buffers[i], cameraUniform.memories[i]
                    );
        }
    }


    void updateUniformBuffer(uint32_t index) {
        // glm::vec3 cameraPosition = glm::vec3(
        //     glm::rotate(glm::mat4(1.0f), app->time * glm::radians(90.0f), WORLD_UP) * glm::vec4(3.0f, 0.0f, 0.0f, 1.0f)
        //         );

        CameraUBO ubo {
            // glm::lookAt(cameraPosition, glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP),
            noclipCam.get_view_matrix(),
            glm::perspective(glm::radians(60.0f), app->swapChainExtent.width / (float) app->swapChainExtent.height, 0.1f, 10.0f),
            noclipCam.position
        };
        // Y coordinate is inverted
        ubo.proj[1][1] *= -1;

        void* data = app->device->mapMemory(cameraUniform.memories[index], 0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        app->device->unmapMemory(cameraUniform.memories[index]);
    }

    virtual void submitGraphics(const vk::CommandBuffer* bufferToSubmit, uint32_t currentFrame) override {
        updateUniformBuffer(currentFrame);
        BaseScreen::submitGraphics(bufferToSubmit, currentFrame);
    }


    virtual void mouseMovementCallback(GLFWwindow* window, double xpos, double ypos) override {
        static double lastxpos = 0, lastypos = 0;

        double xdiff = xpos - lastxpos;
        double ydiff = lastypos - ypos;

#ifdef USE_IMGUI
        if (!ImGui::IsAnyItemActive())
#endif
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            noclipCam.process_mouse(xdiff, ydiff);
            noclipCam.update_vectors();
        }

        lastxpos = xpos;
        lastypos = ypos;
    }

    virtual void update(float delta) override {
        noclipCam.update(delta);
    }

#ifdef USE_IMGUI
    virtual void imgui() override {
    }
#endif

    virtual ~CameraScreen() {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            app->device->destroyBuffer(cameraUniform.buffers[i]);
            app->device->freeMemory(cameraUniform.memories[i]);
        }
    }
};

#endif
