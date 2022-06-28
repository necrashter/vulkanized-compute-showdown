#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
} ubo;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 vel;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragVel;

void main() {
    vec4 worldpos = vec4(pos, 1.0);
    gl_Position = ubo.proj * ubo.view * worldpos;
	gl_PointSize = 8.0;

    fragPos = pos;
	fragVel = vel;
}

