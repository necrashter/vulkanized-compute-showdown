#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
} ubo;

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 vel;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out float fragColor;

void main() {
    const float mass = pos.w;
    const float spriteSize = 0.005 * mass;
    const float screenWidth = ubo.cameraPosition.w;

    vec4 eyePos = ubo.view * vec4(pos.xyz, 1.0); 
    vec4 projectedCorner = ubo.proj * vec4(0.5 * spriteSize, 0.5 * spriteSize, eyePos.z, eyePos.w);
    gl_PointSize = clamp(screenWidth * projectedCorner.x / projectedCorner.w, 1.0, 128.0);
    gl_Position = ubo.proj * eyePos;

    fragColor = vel.w;
}

