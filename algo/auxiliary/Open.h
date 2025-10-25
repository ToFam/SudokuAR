#pragma once

#include "../Algorithm.h"

class Open : public Algorithm
{
public:
    Open();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
