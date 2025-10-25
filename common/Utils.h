#pragma once

#include <filesystem>
#include <string>

namespace Utils
{
std::string loadFile(const std::filesystem::path& path);
}
