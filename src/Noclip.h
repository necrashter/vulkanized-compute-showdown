#ifndef NOCLIP_H
#define NOCLIP_H

#include <GLFW/glfw3.h> // The GLFW header

#include "config.h"

#define GLM_FORCE_RADIANS
// use 0, 1 depth in Vulkan instead of OpenGL's -1 to 1
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


namespace noclip {
    const float max_speed = 8.0f;
    const float max_speed2 = max_speed * max_speed;
    const float acceleration = 32.0f;
    const float deacceleration = -32.0f;
    const float sensitivity = 0.1f;

    class cam {
        public:
            GLFWwindow* window;
            glm::vec3 position;
            glm::vec3 velocity;
            glm::vec3 up;
            glm::vec3 front;
            glm::vec3 right;

            float yaw;
            float pitch;

            cam(GLFWwindow* window,
                    glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
                    float yaw = 0.0f,
                    float pitch = 0.0f) :
                window(window),
                position(position),
                velocity(glm::vec3(0.0f, 0.0f, 0.0f)),
                up(WORLD_UP),
                front(glm::vec3(0.0f, 0.0f, -1.0f)),
                yaw(yaw),
                pitch(pitch) {
                    update_vectors();
                }

            glm::mat4 get_view_matrix() {
                return glm::lookAt(position, position + front, up);
            }

            void update(float delta) {
                glm::vec3 movement(0.0f, 0.0f, 0.0f);

                if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                    movement.z += 1.0f;
                }
                if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                    movement.z -= 1.0f;
                }
                if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                    movement.x -= 1.0f;
                }
                if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                    movement.x += 1.0f;
                }

                glm::vec3 acc(0.0f, 0.0f, 0.0f);
                if (glm::dot(movement, movement) > 1e-6) {
                    acc = glm::mat3x3(right, up, front) * glm::normalize(movement) * noclip::acceleration;

                    velocity += acc * delta;
                    if (glm::dot(velocity, velocity) > noclip::max_speed2) {
                        velocity = glm::normalize(velocity) * noclip::max_speed;
                    }
                    position += velocity * delta;
                } else if (glm::dot(velocity, velocity) > 1e-6) {
                    acc = glm::normalize(velocity) * noclip::deacceleration * delta;
                    if (glm::dot(acc, acc) > glm::dot(velocity, velocity)) {
                        velocity.x = velocity.y = velocity.z = 0.0f;
                        return;
                    }
                    velocity += acc;
                    position += velocity * delta;
                } else {
                    return;
                }
            }

            // NOTE: update vectors after this
            void process_mouse(float xoffset, float yoffset) {
                yaw   += xoffset * noclip::sensitivity;
                pitch += yoffset * noclip::sensitivity;

                if (pitch > 89.0f) {
                    pitch = 89.0f;
                } else if (pitch < -89.0f) {
                    pitch = -89.0f;
                }
            }

            void update_vectors() {
                glm::vec3 new_front(
                        cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
                        sin(glm::radians(pitch)),
                        sin(glm::radians(yaw)) * cos(glm::radians(pitch))
                        );
                front = glm::normalize(new_front);
                right = glm::normalize(glm::cross(front, WORLD_UP));
                up    = glm::normalize(glm::cross(right, front));
            }

            void update_vectors_custom(float yawmod, float pitchmod, glm::vec3 custom_up = WORLD_UP) {
                float finalyaw = glm::radians(yaw) + yawmod;
                float finalpitch = glm::radians(pitch) + pitchmod;
                glm::vec3 new_front(
                        cos(finalyaw) * cos(finalpitch),
                        sin(finalpitch),
                        sin(finalyaw) * cos(finalpitch)
                        );
                front = glm::normalize(new_front);
                right = glm::normalize(glm::cross(front, custom_up));
                up    = glm::normalize(glm::cross(right, front));
            }
    };

}


#endif

