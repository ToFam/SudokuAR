SET (AlgoSrc
    Algorithm.h
    Algorithm.cpp
    AlgorithmSettings.h
    AlgorithmSettings.cpp
    Container.h
    Container.cpp
	LineAlgorithms.h
	LineAlgorithms.cpp
    util.h
    util.cpp
    OCR.h
    OCR.cpp
    
	sudoku/FindPerpLines.h
	sudoku/FindPerpLines.cpp
	sudoku/FitRegularPerpLines.h
	sudoku/FitRegularPerpLines.cpp
    sudoku/HoughTransform.h
    sudoku/HoughTransform.cpp
    sudoku/HoughTransformProb.h
    sudoku/HoughTransformProb.cpp
    sudoku/LineSegmenter.h
    sudoku/LineSegmenter.cpp
    sudoku/LineGrouping.h
    sudoku/LineGrouping.cpp
    sudoku/GridDetect.h
    sudoku/GridDetect.cpp
    sudoku/GridRefine.h
    sudoku/GridRefine.cpp
    sudoku/TemplateMatch.h
    sudoku/TemplateMatch.cpp
    sudoku/TemplateMatchCPU.cpp
    sudoku/TemplateMatchGPU.cpp
    sudoku/TemplateMatchOCVGPU.cpp
    sudoku/TemplateMatchGPUOther.cpp
    sudoku/TemplateMatch.cl
    sudoku/Sudoku.h
    sudoku/Sudoku.cpp
    sudoku/Sudoku.cl
    sudoku/SolutionDisplay.h
    sudoku/SolutionDisplay.cpp
    
    auxiliary/Gray.h
    auxiliary/Gray.cpp
    auxiliary/Blur.h
    auxiliary/Blur.cpp
    auxiliary/Threshold.h
    auxiliary/Threshold.cpp
    auxiliary/GaussianBlur.h
    auxiliary/GaussianBlur.cpp
    auxiliary/Bilateral.h
    auxiliary/Bilateral.cpp
    auxiliary/Open.h
    auxiliary/Open.cpp
    auxiliary/Close.h
    auxiliary/Close.cpp
	auxiliary/Canny.h
	auxiliary/Canny.cpp
	auxiliary/ROI.h
	auxiliary/ROI.cpp
)

add_library(${AlgoTarget}
            ${AlgoSrc}
            auxiliary/Resize.h auxiliary/Resize.cpp)

target_include_directories(${AlgoTarget} PUBLIC ../common)

target_link_libraries(${AlgoTarget}
    PRIVATE
    ${CommonTarget}
    OpenCL::OpenCL
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)
