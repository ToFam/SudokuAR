#pragma once

#include "../Algorithm.h"

class LineGrouping : public Algorithm
{
public:
    LineGrouping();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
