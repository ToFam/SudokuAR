#include "Utils.h"

#include <fstream>
#include <format>

namespace Utils
{
std::string loadFile(const std::filesystem::__cxx11::path &path)
{
    std::ifstream source(path);
    if (!source.is_open())
    {
        throw std::runtime_error(std::format("Cannot open file {}", std::string(path)));
    }

    size_t fileSize = std::filesystem::file_size(path);
    std::string rtn(fileSize, '\0');
    source.read(&rtn[0], fileSize);
    return rtn;
}
}
