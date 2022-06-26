cmake_minimum_required(VERSION 3.13)


project(
    vulkan-project
    VERSION 0.0.1
    DESCRIPTION "Vulkan Project"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g")


# COMPILE OPTIONS
option(USE_IMGUI "Enable Dear Imgui" ON)


#######################################################################
#                              LIBRARIES                              #
#######################################################################

if(USE_IMGUI)
    # add the MuhLib library
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
message(STATUS "GLM found ${glm_FOUND}")

# Vulkan itself
find_package(Vulkan REQUIRED FATAL_ERROR)


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
      ${glslc_executable}
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
