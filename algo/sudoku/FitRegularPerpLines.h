#pragma once

#include "../Algorithm.h"

class FitRegularPerpLines : public Algorithm
{
public:
    FitRegularPerpLines();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
