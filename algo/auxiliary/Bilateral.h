#pragma once

#include "../Algorithm.h"

class Bilateral : public Algorithm
{
public:
    Bilateral();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
