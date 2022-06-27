#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

class ArgParser {
public:
    std::unordered_map<std::string, std::string> args;
    std::vector<std::string> posargs;

    ArgParser(int argc, char** argv);

    inline bool hasArg(const char* const key) {
        return args.find(key) != args.end();
    }

    std::optional<std::string> getArgStr(const char* const key);

    std::optional<bool> getArgBool(const char* const key);

    std::optional<int> getArgInt(const char* const key);
};

#endif
