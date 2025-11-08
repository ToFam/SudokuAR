#include "Test.h"

#include <CLUtil.h>
#include <sudoku/Sudoku.h>

#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;

bool openInputData(std::string file, std::string element, cv::Mat& out)
{
    cv::FileStorage fs(file, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        if (!fs[element].isNone())
        {
            fs[element] >> out;
            return true;
        }
    }

    cout << "Error opening evaluation data file! (" << file << ")" << endl;
    return false;
}

bool saveOutputData(std::string file, std::string element, cv::Mat& data, bool image)
{
    if (image)
    {
        if (cv::imwrite(file, data))
        {
            return true;
        }
    }
    else
    {
        cv::FileStorage fs(file, cv::FileStorage::WRITE);
        if (fs.isOpened())
        {
            fs << element << data;
            return true;
        }
    }

    cout << "Error saving output file! (" << file << ")" << endl;
    return false;
}

bool TestSolver::DoCompute()
{
	cout<<"########################################"<<endl;
    cout<<"Running Sudoku Solver..."<<endl<<endl;
    {
        std::shared_ptr<cv::Mat> in(new cv::Mat()), out(new cv::Mat());
        if (openInputData("eval/solver/SudokuSolver_9_1.yml", "field", *in))
        {
            Sudoku s(9);
            s.setLogLevel(4);
            m_context.initTask(s);

            auto ci(std::make_shared<Container>()), co(std::make_shared<Container>());
            ci->set(in);
            co->set(out);

            s.addContainer(ci);
            s.addContainer(co);


            s.setImplementation(Algorithm::CPU);
            RunAlgorithm(&s, 1, 1, *co, "eval/solver/SudokuSolver_9", "field", false);
        }

    }

	return true;
}

bool TestSolver::RunAlgorithm(Algorithm* algo, int iterationsCPU, int iterationsGPU, Container& outputContainer,
                              const string &outputFilePrefix, const string &outputFileElement, bool outputImage)
{
    std::vector<Algorithm::ImplementationType> supported = algo->supportedImplementations();

    // Compute the golden result.
    cout << "Reference:" << endl;

    bool reference = false;

    for (Algorithm::ImplementationType type : supported)
    {
        std::string outputFileName;
        if (type == Algorithm::ImplementationType::CPU)
        {
            cout << "Computing CPU...";
            outputFileName = "_Result_CPU";
            algo->setIterations(iterationsCPU);
        }
        else if (type == Algorithm::ImplementationType::OPENCV_CPU)
        {
            cout << "Computing OpenCV CPU...";
            outputFileName = "_Result_OCV_CPU";
            algo->setIterations(iterationsCPU);
        }
        else if (type == Algorithm::ImplementationType::OPENCV_GPU)
        {
            cout << "Computing OpenCV GPU...";
            outputFileName = "_Result_OCV_GPU";
            algo->setIterations(iterationsGPU);
        }
        else
        {
            continue;
        }


        algo->setImplementation(type);

        if (algo->exec())
        {
            cout << "DONE average time: " << algo->runtime() << " ms " << std::endl;

            saveOutputData(outputFilePrefix + outputFileName + ((outputImage) ? ".png" : ".yml"), outputFileElement, *outputContainer.get(), outputImage);
        }
        else
        {
            cout << "ERROR!" << endl;
        }

        reference = true;
    }


    if (!reference)
    {
        cout << "Warning: No reference implementation!" << endl;
    }

    // Running the same task on the GPU.
    cout << endl << "Computing GPU result...";

    // Runing the kernel N times. This make the measurement of the execution time more accurate.
    if (!algo->setImplementation(Algorithm::ImplementationType::GPU))
    {
        cout << "ERROR: No GPU implementation!" << endl;
        return false;
    }

    algo->setIterations(iterationsGPU);
    if (algo->exec())
    {
        cout << "DONE average time: " << algo->runtime() << " ms " << std::endl;

        saveOutputData(outputFilePrefix + "_Result_GPU" + ((outputImage) ? ".png" : ".yml"), outputFileElement, *outputContainer.get(), outputImage);
    }
    else
    {
        cout << "ERROR!" << endl;
    }

    if (reference)
    {
        // Validating results.
        if (static_cast<Sudoku*>(algo)->ValidateResults())
        {
            cout << "GOLD TEST PASSED!" << endl;
        }
        else
        {
            cout << "INVALID RESULTS!" << endl;
        }
    }

    return true;
}

