#pragma once

#include <glm/glm.hpp>

namespace glsl {

using namespace glm;

// Unfortunately this doesn't work due to swizzle operations
// #include "hsv.glsl"
vec3 rgb2hsv(vec3 c);
vec3 hsv2rgb(vec3 c);

}
