#pragma once

#include <memory>
#include <vector>
#include <string>

class Algorithm;
class OCR;

std::unique_ptr<Algorithm> createAlgorithm(std::string_view name, const OCR& ocr);

std::vector<std::string> allAlgoNames();
