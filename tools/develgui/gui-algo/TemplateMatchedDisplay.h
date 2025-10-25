#pragma once

#include <Algorithm.h>
#include <OCR.h>

class TemplateMatchedDisplay : public Algorithm
{
public:
    TemplateMatchedDisplay(const OCR& ocr);

    std::vector<ImplementationType> supportedImplementations() const override;
    bool exec() override;

private:
    const OCR& m_ocr;
};
