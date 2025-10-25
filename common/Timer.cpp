#include "Timer.h"

Timer::Timer()
{
    m_cp = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed() const
{
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t - m_cp).count();
}

double Timer::restart()
{
    auto t = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t - m_cp).count();
    m_cp = t;
    return elapsed;
}
