#pragma once

#include "../Algorithm.h"

class FindPerpLines : public Algorithm
{
public:
    FindPerpLines();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
