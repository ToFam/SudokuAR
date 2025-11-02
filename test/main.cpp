#include "CLUtil.h"
//#include "TestTemplateMatch.h"
#include "Test.h"

using namespace std;

int main(int argc, char** argv)
{
    bool success = true;

    CLUtil::CLHandler handler;

    //TestTemplateMatch ttm(handler);
    //ttm.DoCompute();

    TestSolver testSudoku(handler);
    testSudoku.DoCompute();

	return success ? 0 : 1;
}
