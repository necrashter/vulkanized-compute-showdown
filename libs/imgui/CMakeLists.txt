file(GLOB IMGUI_SRCS "*.cpp")

add_library(imgui
    ${IMGUI_SRCS}
)

target_compile_options(imgui PRIVATE -O2)
