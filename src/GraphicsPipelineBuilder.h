#ifndef GRAPHICS_PIPELINE_BUILDER_H
#define GRAPHICS_PIPELINE_BUILDER_H

#include <vulkan/vulkan.hpp>

class GraphicsPipelineBuilder {
public:
    // viewport and scissors are dynamic, hence nullptr (ignored)
    static constexpr auto viewportState = vk::PipelineViewportStateCreateInfo(
            {},
            1, nullptr,
            1, nullptr);

    std::vector<vk::PipelineShaderStageCreateInfo> stages;

    vk::PipelineVertexInputStateCreateInfo vertexInput;
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;

    vk::PipelineRasterizationStateCreateInfo rasterizer;
    vk::PipelineMultisampleStateCreateInfo multisampling;
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    vk::PipelineColorBlendStateCreateInfo colorBlending;

    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstants;
    // After the pipeline is created, this attribute will be set appropriately
    vk::PipelineLayout pipelineLayout;

    vk::PipelineDepthStencilStateCreateInfo depthStencil;

    std::vector<vk::DynamicState> dynamicStates;

    vk::GraphicsPipelineCreateInfo pipelineInfo;

    GraphicsPipelineBuilder():
        stages(),
        vertexInput({}, 0, nullptr, 0, nullptr),
        inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE),
        rasterizer({},
                VK_FALSE, // depth clamp
                VK_FALSE, // rasterizer discard
                vk::PolygonMode::eFill,
                vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise,
                VK_FALSE, {}, {}, {}, // Depth bias settings
                1.0f /* line width */),
        multisampling({}, vk::SampleCountFlagBits::e1, VK_FALSE),
        // Default blending
        colorBlendAttachment(
                VK_TRUE, // blend enable
                // Color blend: src and dst factors, blend operation
                vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
                // Alpha blend: src and dst factors, blend operation
                vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                // Color write mask
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA),
        colorBlending({},
                // LogicOp enable and logic op
                VK_FALSE, vk::LogicOp::eCopy,
                // color blend attachments
                1, &colorBlendAttachment,
                // Blend constants
                {0.0f, 0.0f, 0.0f, 0.0f}),
        depthStencil({},
                VK_TRUE, // depth test enable
                VK_TRUE, // depth test write enable
                vk::CompareOp::eLess,
                VK_FALSE, VK_FALSE),
        dynamicStates {
                vk::DynamicState::eViewport,
                vk::DynamicState::eScissor,
        },
        pipelineInfo({},
                0, nullptr, // stages
                // vertex input
                &vertexInput, &inputAssembly,
                nullptr, // tesselation
                &viewportState, &rasterizer, &multisampling, &depthStencil, &colorBlending,
                nullptr, // dynamicState
                nullptr, // pipelineLayout
                {}, // renderPass
                0, // subpass
                nullptr // base layout handle
                ) {
    }


    // NOTE: take pipelineLayout from attribute after building
    vk::Pipeline build(vk::Device &device) {
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({},
                descriptorSetLayouts.size(), descriptorSetLayouts.data(),
                pushConstants.size(), pushConstants.data()
                );
        try {
            pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
        } catch (vk::SystemError const &err) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        pipelineInfo.stageCount = stages.size();
        pipelineInfo.pStages = stages.data();
        pipelineInfo.layout = pipelineLayout;
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates.size(), dynamicStates.data());
        pipelineInfo.pDynamicState = &dynamicState;

        return device.createGraphicsPipeline(
                nullptr, // cache
                pipelineInfo).value;
    }
};

#endif
