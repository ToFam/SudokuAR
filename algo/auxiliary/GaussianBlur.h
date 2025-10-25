#pragma once

#include "../Algorithm.h"

class GaussianBlur : public Algorithm
{
public:
    GaussianBlur();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
