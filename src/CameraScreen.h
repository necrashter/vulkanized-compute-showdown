#ifndef CAMERA_SCREEN_H
#define CAMERA_SCREEN_H

#include "VulkanBaseApp.h"
#include "Noclip.h"
#include "FrameUniform.h"


struct CameraUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 cameraPosition;
};


class CameraScreen : public BaseScreen {
protected:
    noclip::cam noclipCam;
    float fov = 60.0f;

    FrameUniform cameraUniform;


public:
    CameraScreen(VulkanBaseApp* app):
        BaseScreen(app),
        noclipCam(app->window),
        cameraUniform(app, sizeof(CameraUBO)) {
    }


    void updateUniformBuffer(uint32_t index) {
        // glm::vec3 cameraPosition = glm::vec3(
        //     glm::rotate(glm::mat4(1.0f), app->time * glm::radians(90.0f), WORLD_UP) * glm::vec4(3.0f, 0.0f, 0.0f, 1.0f)
        //         );

        CameraUBO ubo {
            // glm::lookAt(cameraPosition, glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP),
            noclipCam.get_view_matrix(),
            glm::perspective(glm::radians(fov), app->swapChainExtent.width / (float) app->swapChainExtent.height, 0.02f, 100.0f),
            noclipCam.position
        };
        // Y coordinate is inverted
        ubo.proj[1][1] *= -1;

        memcpy(cameraUniform.mappings[index], &ubo, sizeof(ubo));
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
        static bool imguiShowCamera = false;
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("Scene")) {
                ImGui::MenuItem("Camera Settings", NULL, &imguiShowCamera);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (imguiShowCamera) {
            ImGui::Begin("Camera", &imguiShowCamera);
            ImGui::Text("Position %.3f %.3f %.3f", noclipCam.position.x, noclipCam.position.y, noclipCam.position.z);
            ImGui::Text("Yaw/Pitch %.3f %.3f", noclipCam.yaw, noclipCam.pitch);
            ImGui::DragFloat("FOV", &fov, 0.5f, 10, 180, "%.3f");
            ImGui::End();
        }
    }
#endif
};

#endif
