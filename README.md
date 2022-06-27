
# Build

Create build folder:
```sh
$ mkdir bin
$ cd bin
```

Configure with default settings in CMakeLists.txt:
```sh
$ cmake ..
```

Configure with custom settings; enable/disable external dependencies:
```sh
$ cmake -DUSE_IMGUI=ON -DUSE_SHADERC=OFF ..
```
These are the default settings.

Compile with 4 processes:
```sh
$ make -j4
```

Run:
```
$ ./main
```


# Future Work

## Optimizations

- Implement a dedicated transfer queue
- Optimize one shot commands
- More efficient swap chain recreation, see vulkan tut comments
    - Done, not as good as I initially expected, but avoids recompiling shaders
	- Update: It's now as good as I expected. Renderpass doesn't need to be recreated, no need to wait events etc.
- _"Using a UBO this way is not the most efficient way to pass frequently changing values to the shader. A more efficient way to pass a small buffer of data to shaders are push constants. We may look at these in a future chapter."_
    - Push Constants are done as a part of gltf rendering


## VulkanTutorial Conclusion

Not done from tutorial:
- [Multi-sampling](https://vulkan-tutorial.com/en/Multisampling). I don't think it's required.

Ideas from conclusion:
- Push constants
    - I took a detour and implemented model loading with glTF instead of obj. Push constants are implemented to pass model matrix for hierarchical rendering.
- Instanced rendering
- Dynamic uniforms
- Separate images and sampler descriptors
    - Done as a result of implementing glTF support.
- Pipeline cache
- Multi-threaded command buffer generation
- Multiple subpasses
- Compute shaders

