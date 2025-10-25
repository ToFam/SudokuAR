#pragma once

#include "../Algorithm.h"

class Blur : public Algorithm
{
public:
    Blur();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
