#include "VulkanBaseApp.h"
#include "ArgParser.h"
#include "config.h"
#include "log.h"

#include <iostream>


int main(int argc, char** argv) {
    ArgParser argparser(argc, argv);

    printInfo();

    std::function<BaseScreen*(VulkanBaseApp*)> startingScreen = nullptr;
    if (auto arg = argparser.getArgStr("screen")) {
        startingScreen = findScreen(*arg);
        if (startingScreen == nullptr) {
            TLOG("ArgParser") << "Screen not found: " << *arg << std::endl;
        }
    }

    if (auto arg = argparser.getArgBool("validation")) {
        enableValidationLayers = *arg;
        TLOG("ArgParser") << "Explicitly " << (*arg ? "enabling" : "disabling") << " validation layer" << std::endl;
    }

    if (auto arg = argparser.getArgBool("single-queue")) {
        preferSingleQueueFamily = *arg;
        TLOG("ArgParser") << "Prefer single queue family: " << (*arg
                ? "On (Don't prefer dedicated queue families)"
                : "Off (Prefer dedicated queue families)")  << std::endl;
    }

    if (argparser.hasArg("list-gpus")) {
        listGPUs = true;
    }

    selectedGPU = argparser.getArgInt("gpu");

    try {
        VulkanBaseApp app;
        if (startingScreen) app.screen = startingScreen(&app);
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
