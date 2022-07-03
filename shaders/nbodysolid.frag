#version 450

#include "lib/hsv.glsl"

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
} ubo;

layout(set = 0, binding = 1) uniform FrameUBO {
	float colorShift;
} particle;

/* layout(set = 1, binding = 0) uniform sampler2D texSampler; */

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in float fragHue;

layout(location = 0) out vec4 outColor;

const vec3 L = normalize(vec3(1, 2, 1));

void main() {
    const vec4 color = vec4(hsv2rgb(vec3(fragHue + particle.colorShift, 1.0, 1.0)), 1.0f);
    const vec3 N = normalize(fragNormal);
    const vec3 V = normalize(ubo.cameraPosition - fragPos);
    const vec3 H = normalize(L + V);

    vec3 diffuse  = color.rgb * max(0.2, dot(N, L));
    vec3 specular = vec3(0.75f) * pow(max(0, dot(N, H)), 100);

    outColor = vec4(diffuse + specular, 1.0f);
}

