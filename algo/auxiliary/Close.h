#pragma once

#include "../Algorithm.h"

class Close : public Algorithm
{
public:
    Close();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
