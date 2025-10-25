#pragma once

#include "../Algorithm.h"

class HoughTransform : public Algorithm
{
public:
    HoughTransform();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
