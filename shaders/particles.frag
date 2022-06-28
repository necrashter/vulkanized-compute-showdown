#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
} ubo;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 vel;

layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(1, 1, 1, 1.0f);
}

