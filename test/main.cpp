#include "CLUtil.h"
#include "TestTemplateMatch.h"

using namespace std;

int main(int argc, char** argv)
{
    bool success = true;

    CLUtil::CLHandler handler;

    TestTemplateMatch ttm(handler);
    ttm.DoCompute();

	return success ? 0 : 1;
}
