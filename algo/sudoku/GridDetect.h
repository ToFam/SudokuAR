#pragma once

#include "../Algorithm.h"

class GridDetect : public Algorithm
{
public:
    GridDetect();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
