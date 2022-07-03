#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
} ubo;

// Per-vertex
layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNormal;

// Per-instance
layout(location = 4) in vec4 pos;
layout(location = 5) in vec4 vel;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out float fragColor;

void main() {
    vec4 worldpos = vec4(pos.xyz + pos.w*vPos, 1.0);
    gl_Position = ubo.proj * ubo.view * worldpos;

    fragPos = worldpos.xyz;
    fragNormal = vNormal;

    fragColor = vel.w;
}

