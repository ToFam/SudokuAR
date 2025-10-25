#pragma once

#include "../Algorithm.h"

class Gray : public Algorithm
{
public:
    Gray();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
