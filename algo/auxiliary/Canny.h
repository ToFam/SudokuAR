#pragma once

#include "../Algorithm.h"

class Canny : public Algorithm
{
public:
    Canny();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
