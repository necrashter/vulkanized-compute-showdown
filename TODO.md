
# Optimizations

- More efficient swap chain recreation, see vulkan tut comments
- The actual code on the site does not recreate command buffers with swap chain on resize. And commands are recorded each frame.
    - Why do I need to recreate command buffers? It crashes otherwise. Maybe just re-record required.
- _"Using a UBO this way is not the most efficient way to pass frequently changing values to the shader. A more efficient way to pass a small buffer of data to shaders are push constants. We may look at these in a future chapter."_
