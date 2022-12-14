cmake_minimum_required(VERSION 3.13)


project(
    vulkanized-compute-showdown
    VERSION 1.0.0
    DESCRIPTION "Vulkanized Compute Showdown"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g")


# COMPILE OPTIONS
option(USE_IMGUI "Enable Dear Imgui for a nice graphical user interface" ON)
option(USE_LIBKTX "Enable libktx to read GPU-oriented texture format KTX" OFF)

# UNUSED
option(USE_SHADERC "Enable shaderc for compiling GLSL shaders at run-time" OFF)


#######################################################################
#                              LIBRARIES                              #
#######################################################################

# GLSL shaders
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/shaders/lib")

if(USE_IMGUI)
    add_subdirectory(libs/imgui)
    list(APPEND EXTRA_LIBS imgui)
    list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/libs/imgui")
    add_compile_definitions(USE_IMGUI)
endif()

# Tiny GLTF
add_subdirectory(libs/tinygltf)
list(APPEND EXTRA_LIBS tiny_gltf)
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/libs/tinygltf")


# Add GLM
set(glm_DIR libs/glm/cmake/glm)
find_package(glm REQUIRED)

message(STATUS "GLM includes ${GLM_INCLUDE_DIRS}")

# Vulkan itself
find_package(Vulkan REQUIRED FATAL_ERROR)

# shaderc
if(USE_LIBKTX)
    find_library(LIBKTX ktx)
    if (LIBKTX)
        message(STATUS "Found libktx: ${LIBKTX}")
        list(APPEND EXTRA_LIBS ${LIBKTX})
        add_compile_definitions(USE_LIBKTX)
    else()
        message(FATAL_ERROR "libktx was requested but NOT found!")
    endif()
endif()

# shaderc
if(USE_SHADERC)
    find_library(SHADERC shaderc_combined)
    if (SHADERC)
        message(STATUS "Found shaderc: ${SHADERC}")
        list(APPEND EXTRA_LIBS ${SHADERC})
        list(APPEND EXTRA_LIBS pthread)
        add_compile_definitions(USE_SHADERC)
    else()
        message(FATAL_ERROR "shaderc was requested but NOT found!")
    endif()
endif()

#######################################################################
#                               SHADERS                               #
#######################################################################

find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(GLOB SHADERS
  ${SHADER_SOURCE_DIR}/*.vert
  ${SHADER_SOURCE_DIR}/*.frag
  ${SHADER_SOURCE_DIR}/*.comp
  ${SHADER_SOURCE_DIR}/*.geom
  ${SHADER_SOURCE_DIR}/*.tesc
  ${SHADER_SOURCE_DIR}/*.tese
  ${SHADER_SOURCE_DIR}/*.mesh
  ${SHADER_SOURCE_DIR}/*.task
  ${SHADER_SOURCE_DIR}/*.rgen
  ${SHADER_SOURCE_DIR}/*.rchit
  ${SHADER_SOURCE_DIR}/*.rmiss)

add_custom_command(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
  OUTPUT ${SHADER_BINARY_DIR}
  COMMENT "Creating ${SHADER_BINARY_DIR}"
)

foreach(source IN LISTS SHADERS)
  get_filename_component(FILENAME ${source} NAME)
  add_custom_command(
    COMMAND
      ${glslc_executable} -O
      -o ${SHADER_BINARY_DIR}/${FILENAME}.spv
      ${source}
    OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
    DEPENDS ${SHADER_BINARY_DIR}
    COMMENT "Compiling shader: ${FILENAME}"
  )
  list(APPEND SPV_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.spv)
endforeach()

add_custom_target(compile-shaders ALL DEPENDS ${SPV_SHADERS})


#######################################################################
#                           MAIN EXECUTABLE                           #
#######################################################################

# Add the executable
file(GLOB MAIN_SRC "src/*.cpp")
add_executable(main ${MAIN_SRC})

# Add the binary tree to the search path for include files
target_include_directories(main PUBLIC
	"${PROJECT_BINARY_DIR}"
	"${PROJECT_SOURCE_DIR}/src"
	${GLM_INCLUDE_DIRS}
	${GLFW_INCLUDE_DIRS}
	${Vulkan_INCLUDE_DIRS}
	${EXTRA_INCLUDES}
	)

target_link_libraries(main PUBLIC
	glfw ${GLFW_LIBRARIES}
	${Vulkan_LIBRARIES}
	${EXTRA_LIBS}
	)

install(TARGETS main DESTINATION bin)

