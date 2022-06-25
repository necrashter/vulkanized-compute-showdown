
# Build

Create build folder:
```sh
$ mkdir bin
$ cd bin
```

Configure without muhlib:
```sh
$ cmake ..
```
With muhlib:
```sh
$ cmake .. -DUSE_MUHLIB=ON
```

Compile & run:
```sh
$ make
$ ./App
```


# Future Work

## Optimizations

- More efficient swap chain recreation, see vulkan tut comments
- The actual code on the site does not recreate command buffers with swap chain on resize. And commands are recorded each frame.
    - Why do I need to recreate command buffers? It crashes otherwise. Maybe just re-record required.
- _"Using a UBO this way is not the most efficient way to pass frequently changing values to the shader. A more efficient way to pass a small buffer of data to shaders are push constants. We may look at these in a future chapter."_
- Implement a dedicated transfer queue
- Optimize one shot commands


## VulkanTutorial Conclusion

Not done from tutorial:
- [Multi-sampling](https://vulkan-tutorial.com/en/Multisampling). I don't think it's required.

Ideas from conclusion:
- Push constants
    - I took a detour and implemented model loading with glTF instead of obj. Push constants can be implemented to pass model matrix for hierarchical rendering.
- Instanced rendering
- Dynamic uniforms
- Separate images and sampler descriptors
    - Done as a result of implementing glTF support.
- Pipeline cache
- Multi-threaded command buffer generation
- Multiple subpasses
- Compute shaders

