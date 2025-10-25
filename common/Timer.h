#pragma once

#include <chrono>

class Timer
{
public:
    Timer();

    double elapsed() const;
    double restart();

private:
    std::chrono::high_resolution_clock::time_point m_cp;
};

