#version 450

layout(set = 0, binding = 1) uniform FrameUBO {
	float colorShift;
} particle;

layout (binding = 2) uniform sampler1D hueMap;
layout (binding = 3) uniform sampler2D particleTexture;

layout(location = 0) in float color;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(
        texture(hueMap, color + particle.colorShift).rgb,
        texture(particleTexture, gl_PointCoord).r
        );
}

