#pragma once

#include "../Algorithm.h"

class LineSegmenter : public Algorithm
{
public:
    LineSegmenter();

public:
    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;
};
