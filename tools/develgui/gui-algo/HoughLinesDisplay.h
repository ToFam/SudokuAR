#pragma once

#include <Algorithm.h>

class HoughLinesDisplay : public Algorithm
{
public:
    HoughLinesDisplay();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
