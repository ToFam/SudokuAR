#include "AlgoFactory.h"

#include <sudoku/HoughTransform.h>
#include <sudoku/HoughTransformProb.h>
#include <sudoku/FindPerpLines.h>
#include <sudoku/FitRegularPerpLines.h>
#include <sudoku/GridDetect.h>
#include <sudoku/TemplateMatch.h>
#include <sudoku/Sudoku.h>
#include <sudoku/SolutionDisplay.h>
#include <sudoku/LineSegmenter.h>
#include <sudoku/LineGrouping.h>
#include <sudoku/GridRefine.h>

#include <auxiliary/ROI.h>
#include <auxiliary/Gray.h>
#include <auxiliary/Resize.h>
#include <auxiliary/Blur.h>
#include <auxiliary/Canny.h>
#include <auxiliary/Threshold.h>
#include <auxiliary/GaussianBlur.h>
#include <auxiliary/Bilateral.h>
#include <auxiliary/Open.h>
#include <auxiliary/Close.h>

#include <gui-algo/HoughLinesDisplay.h>
#include <gui-algo/GridDetectDisplay.h>
#include <gui-algo/TemplateMatchedDisplay.h>

std::vector<std::string> allAlgoNames()
{
    return {"Gray" , "ROI" , "Resize", "Canny" , "Threshold" , "Blur" , "Bilateral" , "GaussianBlur"
           , "Open" , "Close"
           , "HoughTransform" , "HoughTransformProb" , "FindPerpLines" , "FitRegularPerpLines"
           , "LineSegmenter" , "LineGrouping"
           , "GridDetect" , "GridRefine"
           , "TemplateMatch" , "Sudoku" , "SolutionDisplay"
            , "HoughLinesDisplay" , "GridDetectDisplay" , "TemplateMatchedDisplay"};
}

std::unique_ptr<Algorithm> createAlgorithm(std::string_view name, const OCR& ocr)
{
    if (name == "ROI")
    {
        return std::make_unique<ROI>();
    }
    if (name == "Gray")
    {
        return std::make_unique<Gray>();
    }
    if (name == "Resize")
    {
        return std::make_unique<Resize>();
    }
    if (name == "Canny")
    {
        return std::make_unique<Canny>();
    }
    if (name == "Threshold")
    {
        return std::make_unique<Threshold>();
    }
    if (name == "Blur")
    {
        return std::make_unique<Blur>();
    }
    if (name == "GaussianBlur")
    {
        return std::make_unique<GaussianBlur>();
    }
    if (name == "Bilateral")
    {
        return std::make_unique<Bilateral>();
    }
    if (name == "Open")
    {
        return std::make_unique<Open>();
    }
    if (name == "Close")
    {
        return std::make_unique<Close>();
    }
    if (name == "HoughLinesDisplay")
    {
        return std::make_unique<HoughLinesDisplay>();
    }
    if (name == "GridDetectDisplay")
    {
        return std::make_unique<GridDetectDisplay>();
    }
    if (name == "HoughTransform")
    {
        return std::make_unique<HoughTransform>();
    }
    if (name == "HoughTransformProb")
    {
        return std::make_unique<HoughTransformProb>();
    }
    if (name == "FindPerpLines")
    {
        return std::make_unique<FindPerpLines>();
    }
    if (name == "FitRegularPerpLines")
    {
        return std::make_unique<FitRegularPerpLines>();
    }
    if (name == "GridDetect")
    {
        return std::make_unique<GridDetect>();
    }
    if (name == "LineSegmenter")
    {
        return std::make_unique<LineSegmenter>();
    }
    if (name == "TemplateMatch")
    {
        return std::make_unique<TemplateMatch>(ocr);
    }
    if (name == "Sudoku")
    {
        return std::make_unique<Sudoku>(9);
    }
    if (name == "SolutionDisplay")
    {
        return std::make_unique<SolutionDisplay>(ocr);
    }
    if (name == "LineGrouping")
    {
        return std::make_unique<LineGrouping>();
    }
    if (name == "GridRefine")
    {
        return std::make_unique<GridRefine>();
    }
    if (name == "TemplateMatchedDisplay")
    {
        return std::make_unique<TemplateMatchedDisplay>(ocr);
    }
    return nullptr;
}
