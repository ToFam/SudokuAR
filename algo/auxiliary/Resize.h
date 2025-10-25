#pragma once

#include "../Algorithm.h"

class Resize : public Algorithm
{
public:
    Resize();

    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
