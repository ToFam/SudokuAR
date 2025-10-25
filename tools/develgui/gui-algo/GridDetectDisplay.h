#pragma once

#include <Algorithm.h>

class GridDetectDisplay : public Algorithm
{
public:
    GridDetectDisplay();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
