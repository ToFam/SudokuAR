#pragma once

namespace CLUtil
{
class CLHandler;
}

class TestTemplateMatch
{
public:
    TestTemplateMatch(const CLUtil::CLHandler& handler);

    bool DoCompute();

private:
    const CLUtil::CLHandler& m_context;
};

