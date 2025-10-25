#pragma once

#include "../Algorithm.h"

class ROI : public Algorithm
{
public:
    ROI();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
