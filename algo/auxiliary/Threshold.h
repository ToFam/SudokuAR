#pragma once

#include "../Algorithm.h"

class Threshold : public Algorithm
{
public:
    Threshold();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
