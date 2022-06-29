#version 450

#include "hsv.glsl"

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
} ubo;

layout(set = 0, binding = 1) uniform FrameUBO {
	float colorShift;
} particle;

layout(location = 0) in vec3 pos;
layout(location = 1) in float color;

layout(location = 0) out vec4 outColor;

void main() {
	vec2 centerVector = gl_PointCoord - vec2(0.5, 0.5);
	float opacity = 0.1f * (1.0f - (dot(centerVector, centerVector) / 0.25f));
	outColor = vec4(hsv2rgb(vec3(color + particle.colorShift, 1.0, 1.0)), opacity);
}

