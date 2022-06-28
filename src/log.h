#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "config.h"

namespace {
    inline std::string getProgramInfoStr() {
        std::stringstream ss;
        ss << ProgramInfo.name << ' '
           << "(Version " << ProgramInfo.version.major << '.'
           << ProgramInfo.version.minor << '.' << ProgramInfo.version.patch << ")\n";
#ifdef USE_IMGUI
        ss << " + Compiled with ImGUI\n";
#else
        ss << " - Compiled without ImGUI\n";
#endif
        return ss.str();
    }
}

const std::string ProgramInfoStr = getProgramInfoStr();

inline void printInfo() {
    std::cout << ProgramInfoStr;
    std::cout << std::endl;
}


inline void printTimeTag(){
    auto now = std::chrono::system_clock::now();
    std::time_t cnow = std::chrono::system_clock::to_time_t(now);
    auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    auto fraction = now - seconds;

    std::string s(64, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&cnow));
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(fraction);
    std::cout << '[';
    std::cout << s << '.';
    std::cout << std::setfill('0') << std::setw(3) << milliseconds.count();
    std::cout << ']';
}

inline void printTimeTagStderr(){
    auto now = std::chrono::system_clock::now();
    std::time_t cnow = std::chrono::system_clock::to_time_t(now);
    auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    auto fraction = now - seconds;

    std::string s(64, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&cnow));
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(fraction);
    std::cout << '[';
    std::cout << s << '.';
    std::cout << std::setfill('0') << std::setw(3) << milliseconds.count();
    std::cout << ']';
}

#define TLOG(s) printTimeTag(); std::cout << '[' << s << "] "
#define TERR(s) printTimeTagStderr(); std::cerr << '[' << s << "] "
