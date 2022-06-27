#include "ArgParser.h"

ArgParser::ArgParser(int argc, char** argv) {
    for (int i = 0; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.length() > 2 && arg[0] == '-' && arg[1] == '-') {
            size_t eq = arg.find('=', 2);
            if (eq == std::string::npos) {
                args[arg.substr(2)] = "";
            } else {
                args[arg.substr(2, eq-2)] = arg.substr(eq+1);
            }
        } else {
            posargs.push_back(arg);
        }
    }
}

std::optional<std::string> ArgParser::getArgStr(const char* const key) {
    auto it = args.find(key);
    if (it == args.end()) return std::nullopt;
    return it->second;
}

std::optional<bool> ArgParser::getArgBool(const char* const key) {
    auto it = args.find(key);
    if (it == args.end()) return std::nullopt;
    if (it->second == "off" || it->second == "false") {
        return false;
    }
    return true;
}

std::optional<int> ArgParser::getArgInt(const char* const key) {
    auto it = args.find(key);
    if (it == args.end()) return std::nullopt;
    return stoi(it->second);
}
