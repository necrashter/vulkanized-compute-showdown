#ifndef CONFIG_H
#define CONFIG_H

const struct {
    const char* name = "Vulkanized Compute Showdown";
    struct {
        const int major = 1;
        const int minor = 0;
        const int patch = 0;
    } version;
} ProgramInfo;


#define GLM_FORCE_RADIANS
// use 0, 1 depth in Vulkan instead of OpenGL's -1 to 1
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

const glm::vec3 WORLD_UP(0.0f, 1.0f, 0.0f);

const int WIDTH = 1280;
const int HEIGHT = 720;

const int MAX_FRAMES_IN_FLIGHT = 2;

// Logs the current time with each message
// #define LOG_TIME

#endif
