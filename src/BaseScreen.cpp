#include "BaseScreen.h"
#include "VulkanBaseApp.h"


void BaseScreen::submitGraphics(const vk::CommandBuffer* bufferToSubmit, uint32_t currentFrame) {
    vk::Semaphore waitSemaphores[] = {
        app->imageAvailableSemaphores[currentFrame]
    };
    vk::PipelineStageFlags waitStages[] = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput
    };
    vk::Semaphore signalSemaphores[] = {
        app->renderFinishedSemaphores[currentFrame]
    };

    vk::SubmitInfo submitInfo(
            std::size(waitSemaphores), waitSemaphores, waitStages,
            1, bufferToSubmit,
            std::size(signalSemaphores), signalSemaphores);

    try {
        app->graphicsQueue.submit(submitInfo, app->inFlightFences[currentFrame]);
    } catch (vk::SystemError const &err) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }
}

#include "ModelViewScreen.h"
#include "EmitterScreen.h"

const std::vector<std::pair<std::string, std::function<BaseScreen*(VulkanBaseApp*)>>> 
screenCreators = {
    {"ModelView", [](VulkanBaseApp* app) { return new ModelViewScreen(app); } },
    {"Emitter", [](VulkanBaseApp* app) { return new EmitterScreen(app); } },
};

std::function<BaseScreen*(VulkanBaseApp*)> findScreen(std::string& query) {
    std::function<BaseScreen*(VulkanBaseApp*)> f = nullptr;
    for (auto s : screenCreators) {
        if (s.first == query) {
            f = s.second;
            break;
        }
    }
    return f;
}
