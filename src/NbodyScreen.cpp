#include "NbodyScreen.h"
#include "glsl.h"
#include <random>

namespace {
    struct ComputeUBO {
        alignas(4) glm::uint32 particleCount;
        alignas(4) glm::float32 delta;
    };

    struct FrameUBO {
        alignas(4) glm::float32 colorShift;
    };

    struct Particle {
        alignas(16) glm::vec4 pos;
        alignas(16) glm::vec4 vel;
    };

    constexpr vk::VertexInputBindingDescription vertexBindingDescription = {
        0, // binding
        sizeof(Particle), // stride
        vk::VertexInputRate::eVertex,
    };
    constexpr vk::VertexInputAttributeDescription vertexAttributeDescriptions[] = {
        vk::VertexInputAttributeDescription(
                0, 0, // location and binding
                vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, pos)),

        vk::VertexInputAttributeDescription(
                1, 0, // location and binding
                vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, vel)),
    };

    std::vector<Particle> generateShaderData(uint32_t count) {
        std::default_random_engine randomEngine((unsigned)time(nullptr));
        std::normal_distribution<float> randomDist(0.0f, 1.0f);

        std::vector<Particle> particles(count);

        uint32_t i = 0;
        for (uint32_t ai = 0; ai < std::size(NbodyScreen::attractors); ++ai) {
            auto& attractor = NbodyScreen::attractors[ai];
            float color = ai / (float) std::size(NbodyScreen::attractors);
            // First particle
            particles[i].pos = glm::vec4(attractor*1.5f, 9000);
            particles[i].vel = glm::vec4(0.0f);
            for (uint32_t j = 1; j < NbodyScreen::particlesPerAttractor; ++j) {
                glm::vec3 relativePos(
                        randomDist(randomEngine),
                        randomDist(randomEngine),
                        randomDist(randomEngine)
                        );
                particles[i].pos = glm::vec4(
                        attractor + relativePos,
                        // mass
                        (randomDist(randomEngine) * 0.5f + 0.5f) * 75.0f
                        );
                glm::vec3 angular = glm::vec3(0.5f, 1.5f, 0.5f) * (((ai % 2) == 0) ? 1.0f : -1.0f);
                glm::vec3 velocity = glm::cross(relativePos, angular) + glm::vec3(randomDist(randomEngine), randomDist(randomEngine), randomDist(randomEngine) * 0.025f);
                particles[i].vel = glm::vec4(velocity, color);

                ++i;
            }
        }
        return particles;
    }

    const char* fragmentShaders[] = {
#ifdef USE_LIBKTX
        "shaders/nbodyKTX.frag.spv",
#endif
        "shaders/nbody.frag.spv",
    };

#ifdef USE_IMGUI
    const char* fragmentShaderNames[] = {
#ifdef USE_LIBKTX
        "Textures & LUT",
#endif
        "Procedural",
    };
#endif

    int selectedFragmentShader = 0;

    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupSize;

    const char* computeShaders[] = {
        "shaders/nbodyp1naive.comp.spv",
        "shaders/nbodyp1shared.comp.spv",
    };

#ifdef USE_IMGUI
    const char* computeShaderNames[] = {
        "Naive Method",
        "Shared Cache",
    };
#endif

    int selectedComputeShader = 1;

    const char* description =
        "N-body simulation approximates the motion of particles under the influence gravity. "
        "At each frame, the simulation is advanced using 2 compute shader passes.\n\n"
        "In the first pass, gravitational force between all bodies are computed to find the force "
        "inflicted on each particle, and the forces are applied to update the velocity. "
        "This pass has a computational complexity of O(n^2).\n\n"
        "In the second pass, the positions are updated using Euler integration. This should "
        "be done in a separate pass for synchronization.\n\n"
        "A shared memory in compute shader can be used to accelerate the memory access."
        "";
}



NbodyScreen::NbodyScreen(VulkanBaseApp* app):
    CameraScreen(app), compute(nullptr), graphicsUniform(nullptr)
{
#ifdef USE_LIBKTX
    huesTexture.load(app, "../assets/hues.ktx");
    particleTexture.load(app, "../assets/particle.ktx");
#endif
    auto limits = app->physicalDevice.getProperties().limits;
    maxComputeSharedMemorySize = (uint32_t)(limits.maxComputeSharedMemorySize / sizeof(glm::vec4));
    maxComputeWorkGroupSize = limits.maxComputeWorkGroupSize[0];

    noclipCam.position = glm::vec3(-6*1.7320508075688772, 6, 0);
    noclipCam.pitch = -30;
    noclipCam.yaw = 0.0;
    noclipCam.update_vectors();

    buildPipeline();

#ifndef USE_IMGUI
    std::cout << "\nDESCRIPTION\n" << description << '\n' << std::endl;
#endif
}


void NbodyScreen::buildPipeline() {
    compute = new ComputeSystem(app);
    graphicsUniform = new FrameUniform(app, sizeof(FrameUBO));
    prepareGraphicsPipeline();
    prepareComputePipeline();
}

////////////////////////////////////////////////////////////////////////
//                         GRAPHICS PIPELINE                          //
////////////////////////////////////////////////////////////////////////


void NbodyScreen::prepareGraphicsPipeline() {
    // Create Descriptor Pool
    // ---------------------------------------------------------------

    vk::DescriptorPoolSize poolSizes[] = {
        // UBO
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT * 2),
#ifdef USE_LIBKTX
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT * 2),
#endif
    };
    vk::DescriptorPoolCreateInfo poolCreateInfo({},
            MAX_FRAMES_IN_FLIGHT,
            std::size(poolSizes), poolSizes);
    try {
        graphics.descriptorPool = app->device->createDescriptorPool(poolCreateInfo);
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    // Create Descriptor Set Layout
    // ---------------------------------------------------------------

    vk::DescriptorSetLayoutBinding bindings[] = {
        vk::DescriptorSetLayoutBinding(
            0, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            ),
        vk::DescriptorSetLayoutBinding(
            1, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            ),
#ifdef USE_LIBKTX
        vk::DescriptorSetLayoutBinding(
            2, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding(
            3, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment),
#endif
    };

    try {
        graphics.descriptorSetLayout = app->device->createDescriptorSetLayout({{}, std::size(bindings), bindings});
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create Descriptor Sets
    // ---------------------------------------------------------------

    try {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, graphics.descriptorSetLayout);
        graphics.descriptorSets = app->device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
                    graphics.descriptorPool,
                    static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
                    layouts.data()));
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorBufferInfo bufferInfo(cameraUniform.buffers[i], 0, sizeof(CameraUBO));
        vk::DescriptorBufferInfo frameUniformInfo(graphicsUniform->buffers[i], 0, sizeof(FrameUBO));
#ifdef USE_LIBKTX
        vk::DescriptorImageInfo huesImageInfo(huesTexture.sampler, huesTexture.view,
                vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo particleImageInfo(particleTexture.sampler, particleTexture.view,
                vk::ImageLayout::eShaderReadOnlyOptimal);
#endif

        vk::WriteDescriptorSet descriptorWrites[] = {
            // UBO
            vk::WriteDescriptorSet(
                    graphics.descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer,
                    nullptr, // image info
                    &bufferInfo
                    ),
            vk::WriteDescriptorSet(
                    graphics.descriptorSets[i], 1, 0, 1, vk::DescriptorType::eUniformBuffer,
                    nullptr, // image info
                    &frameUniformInfo
                    ),
#ifdef USE_LIBKTX
            vk::WriteDescriptorSet(
                    graphics.descriptorSets[i], 2, 0, 1,
                    vk::DescriptorType::eCombinedImageSampler,
                    &huesImageInfo,
                    nullptr
                    ),
            vk::WriteDescriptorSet(
                    graphics.descriptorSets[i], 3, 0, 1,
                    vk::DescriptorType::eCombinedImageSampler,
                    &particleImageInfo,
                    nullptr
                    ),
#endif
        };
        app->device->updateDescriptorSets(std::size(descriptorWrites), descriptorWrites, 0, nullptr);
    }

    // Create Graphics Pipeline
    // ---------------------------------------------------------------

    GraphicsPipelineBuilder pipelineBuilder;

    auto vertShaderModule = app->createShaderModule(readBinaryFile("shaders/nbody.vert.spv"));
    auto fragShaderModule = app->createShaderModule(readBinaryFile(fragmentShaders[selectedFragmentShader]));

    pipelineBuilder.stages = { 
        {
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eVertex,
            *vertShaderModule,
            "main"
        }, 
        {
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eFragment,
            *fragShaderModule,
            "main"
        } 
    };

    // Additive blend
    pipelineBuilder.colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    pipelineBuilder.colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    pipelineBuilder.colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOne;
    pipelineBuilder.colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
    pipelineBuilder.colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    pipelineBuilder.colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eOne;
    // No depth test
    pipelineBuilder.depthStencil.depthTestEnable = VK_FALSE;
    pipelineBuilder.depthStencil.depthWriteEnable = VK_FALSE;

    pipelineBuilder.rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    pipelineBuilder.inputAssembly.topology = vk::PrimitiveTopology::ePointList;

    pipelineBuilder.vertexInput.vertexBindingDescriptionCount = 1;
    pipelineBuilder.vertexInput.pVertexBindingDescriptions = &vertexBindingDescription;
    pipelineBuilder.vertexInput.vertexAttributeDescriptionCount = std::size(vertexAttributeDescriptions);
    pipelineBuilder.vertexInput.pVertexAttributeDescriptions = vertexAttributeDescriptions;

    pipelineBuilder.descriptorSetLayouts = { graphics.descriptorSetLayout };

    pipelineBuilder.pipelineInfo.renderPass = app->renderPass;

    try {
        graphics.pipeline = pipelineBuilder.build(app->device.get());
        graphics.pipelineLayout = pipelineBuilder.pipelineLayout;
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }


    // Semaphore
    graphics.sem = app->device->createSemaphore({});
}



void NbodyScreen::prepareComputePipeline() {
    auto shaderData = generateShaderData(maxParticleCount);

    computeSSBO = compute->createShaderStorage(shaderData.data(), sizeof(Particle) * shaderData.size());
    computeUBO = compute->createUniformBuffer(sizeof(ComputeUBO));
    compute->finalizeLayout();

    {
        struct SpecializationData {
            float gravity;
            float power;
            float soften;
            uint32_t localSize;
        } specializationData;

        vk::SpecializationMapEntry specializationMapEntries [] = {
            vk::SpecializationMapEntry(0, offsetof(SpecializationData, gravity), sizeof(float)),
            vk::SpecializationMapEntry(1, offsetof(SpecializationData, power), sizeof(float)),
            vk::SpecializationMapEntry(2, offsetof(SpecializationData, soften), sizeof(float)),
            vk::SpecializationMapEntry(3, offsetof(SpecializationData, localSize), sizeof(uint32_t)),
        };

        specializationData.localSize = std::min(workgroupSize, maxComputeSharedMemorySize);
        specializationData.gravity = gravity;
        specializationData.power = power;
        specializationData.soften = soften;

        vk::SpecializationInfo specializationInfo(
                std::size(specializationMapEntries),
                specializationMapEntries,
                sizeof(SpecializationData),
                &specializationData
                );

        compute->addPipeline(readBinaryFile(computeShaders[selectedComputeShader]), "main", &specializationInfo);
        compute->addPipeline(readBinaryFile("shaders/nbodyp2.comp.spv"), "main", &specializationInfo);
    }

    compute->recordCommands(maxParticleCount / workgroupSize, 1, 1);

    // kickstart
    compute->signalSemaphore();
}




void NbodyScreen::recordRenderCommands(vk::RenderPassBeginInfo renderPassInfo, vk::CommandBuffer commandBuffer, uint32_t index) {
    uint32_t graphicsQindex = app->queueFamilyIndices.graphics;
    uint32_t computeQindex = compute->queueIndex;

    // Compute shader barrier
    if (graphicsQindex != computeQindex) {
        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eVertexInput,
                {},
                0, nullptr,
                compute->graphicsAcquireBarriers[index].size(), compute->graphicsAcquireBarriers[index].data(),
                0, nullptr);
    }

    glm::vec3 bgColor = glsl::hsv2rgb(glm::vec3(colorShift, 0.25f, bgBrightness));
    vk::ClearValue clearValues[] = {
        vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{
                    bgColor.r, bgColor.r, bgColor.b,
                    1.0f })),
            vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0)),
    };
    renderPassInfo.pClearValues = clearValues;
    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, 1, &graphics.descriptorSets[index], 0, nullptr);
    vk::DeviceSize offsets[] = {0};
    commandBuffer.bindVertexBuffers(0, 1, computeSSBO, offsets);
    commandBuffer.draw(particleCount, 1, 0, 0);

    app->renderUI(commandBuffer);
    commandBuffer.endRenderPass();

    // Release barrier
    if (graphicsQindex != computeQindex) {
        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eVertexInput,
                vk::PipelineStageFlagBits::eComputeShader,
                {},
                0, nullptr,
                compute->graphicsReleaseBarriers[index].size(), compute->graphicsReleaseBarriers[index].data(),
                0, nullptr);
    }
}


void NbodyScreen::update(float delta) {
    CameraScreen::update(delta);
}


void NbodyScreen::submitGraphics(const vk::CommandBuffer* bufferToSubmit, uint32_t currentFrame) {
    updateUniformBuffer(currentFrame);
    {
        ComputeUBO* computeUbo = (ComputeUBO*)computeUBO->mappings[currentFrame];
        computeUbo->particleCount = particleCount;
        computeUbo->delta = app->delta * timeMultiplier;

        FrameUBO* ubo = (FrameUBO*)graphicsUniform->mappings[currentFrame];
        ubo->colorShift = colorShift;
    }

    compute->submitSeqGraphicsCompute(bufferToSubmit, currentFrame, graphics.sem);
}


#ifdef USE_IMGUI
void NbodyScreen::imgui() {
    static bool showParticleSettings = true;
    CameraScreen::imgui();
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Particle Settings", NULL, &showParticleSettings);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (showParticleSettings) {
        if (ImGui::Begin("N-Body Simulation", &showParticleSettings)) {
            ImGui::PushItemWidth(-114);

            if (ImGui::CollapsingHeader("Description")) {
                ImGui::TextWrapped("%s", description);
            }

            if (ImGui::CollapsingHeader("Real-Time Settings")) {
                ImGui::DragFloat("BG Brightness", &bgBrightness, 0.005f, 0.0f, 1.0f, "%.3f");
                ImGui::DragFloat("Color Shift", &colorShift, 0.005f, 0.0f, 1.0f, "%.3f");

                ImGui::Separator();

                ImGui::DragFloat("Time Multiplier", &timeMultiplier, 0.005f, 0.0f, 10.0f, "%.3f");

                ImGui::Separator();

                ImGui::DragInt("Particle Count", (int*) &particleCount, 100, 1, maxParticleCount);

                if (ImGui::Button("Restart")) {
                }
            }

            if (ImGui::CollapsingHeader("Pipeline Settings")) {
                ImGui::Combo("Fragment Shader", &selectedFragmentShader, fragmentShaderNames, std::size(fragmentShaderNames));
                ImGuiTooltip( 
                        "Fragment shader can sample color from textures (faster)"
                        " or generate them procurdurally at each function call."
                        );

#ifndef USE_LIBKTX
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255,25,25,255));
                ImGui::TextWrapped(
                        "WARNING: The program is compiled without KTX support. "
                        "Textures and LUTs are not available in fragment shader.");
                ImGui::PopStyleColor();
#endif

                ImGui::Combo("Compute Shader", &selectedComputeShader, computeShaderNames, std::size(computeShaderNames));

                ImGui::Separator();

                ImGui::TextUnformatted("Specialization Constants for Compute Shader");

                ImGui::DragFloat("Gravity", &gravity, 0.0002f, 0.0f, 1.0f, "%.4f");
                ImGui::DragFloat("Power", &power, 0.05f, 0.0f, 10.0f, "%.2f");
                ImGui::DragFloat("Soften", &soften, 0.005f, 0.0f, 10.0f, "%.3f");

                uint32_t maxWorkGroup = maxComputeWorkGroupSize;
                if (selectedComputeShader == 1) {
                    maxWorkGroup = std::min(maxWorkGroup, maxComputeSharedMemorySize);
                }
                if (maxWorkGroup < workgroupSize) workgroupSize = maxWorkGroup;
                ImGui::DragInt("Workgroup Size", (int*) &workgroupSize, 8, 8, maxWorkGroup);
                ImGuiTooltip( 
                        "Number of threads in a work group.\n"
                        "Also equal to the amount of shared data in the first compute shader pass (if enabled)."
                        );

                ImGui::Text(
                        "Max work group size: %d\nMax shared data: %d",
                        maxComputeWorkGroupSize, maxComputeSharedMemorySize);
                ImGuiTooltip("Limits of the current GPU");

                ImGui::Separator();

                if (ImGui::Button("Rebuild Pipeline")) {
                    app->device->waitIdle();
                    pipelineCleanup();
                    buildPipeline();
                }
                ImGuiTooltip("Recreate the whole pipeline to apply new settings.");
            }

            ImGui::PopItemWidth();
        }
        ImGui::End();
    }
}
#endif


void NbodyScreen::pipelineCleanup() {
    // Graphics pipeline cleanup
    app->device->destroyPipeline(graphics.pipeline);
    app->device->destroyPipelineLayout(graphics.pipelineLayout);
    app->device->destroyDescriptorPool(graphics.descriptorPool);
    app->device->destroyDescriptorSetLayout(graphics.descriptorSetLayout);

    app->device->destroySemaphore(graphics.sem);

    if (compute) {
        delete compute;
        compute = nullptr;
    }

    if (graphicsUniform) {
        delete graphicsUniform;
        graphicsUniform = nullptr;
    }
}


NbodyScreen::~NbodyScreen() {
    pipelineCleanup();

#ifdef USE_LIBKTX
    // textures
    huesTexture.cleanup(app);
    particleTexture.cleanup(app);
#endif
}

