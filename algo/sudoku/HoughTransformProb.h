#pragma once

#include "../Algorithm.h"

class HoughTransformProb : public Algorithm
{
public:
    HoughTransformProb();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
