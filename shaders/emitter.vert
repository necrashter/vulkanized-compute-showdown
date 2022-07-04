#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
} ubo;

layout(location = 0) in vec4 pos;
layout(location = 1) in vec3 vel;
layout(location = 2) in float color;

layout(location = 0) out float fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * vec4(pos.xyz, 1.0);
    gl_PointSize = pos.w;

    fragColor = color;
}

