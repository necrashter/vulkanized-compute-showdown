# Vulkanized Compute Showdown

See [blog post](https://necrashter.github.io/ceng469/project/final) for more information about the project.


## Build

The application has been developed and tested on Ubuntu 20.04.
System-wide installations of GLM and Vulkan SDK are required, alongside the `glslc` executable for compiling shaders (should be included in Vulkan SDK).

Create build folder:
```sh
$ mkdir bin
$ cd bin
```

Configure with the default settings:
```sh
$ cmake ..
```

Alternatively, configure with custom settings, e.g., enable the optional KTX library:
```sh
$ cmake -DUSE_LIBKTX=ON ..
```
For this, [KTX Software](https://github.com/KhronosGroup/KTX-Software) must be installed system-wide.
Simply download the `.deb` package from [this link](https://github.com/KhronosGroup/KTX-Software/releases/tag/v4.0.0) and install it.

For all configuration options, please see `CMakeLists.txt`.

After configuring with CMake, compile with 4 processes:
```sh
$ make -j4
```

Run the program:
```sh
$ ./main
```
The working directory must be `bin` so that the program can load the assets correctly.


## Command-Line Arguments

The application contains an user interface implemented with ImGui, which can be disabled with the configuration argument `-DUSE_IMGUI=OFF`, as well as a basic command line interface.

```sh
# List the available GPUs on the system
$ ./main --list-gpus
# Start with the GPU at index 1
$ ./main --gpu=1
# Start with the GPU at index 1 and disable the validation layer
$ ./main --gpu=1 --validation=off
# Argument order doesn't matter
$ ./main --validation=off --gpu=1
# Start with the rigid body screen. Available screens are: "Emitter", "Nbody", "Rigid"
$ ./main --screen=Rigid
$ ./main --screen=Nbody
$ ./main --screen=Emitter
```

