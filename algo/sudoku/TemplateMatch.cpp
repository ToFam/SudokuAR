#include "TemplateMatch.h"

#include <opencv2/opencv.hpp>
#include <vector>

TemplateMatch::TemplateMatch(const OCR& ocr) : Algorithm("TemplateMatch"),
    m_ocr(ocr), m_program(nullptr), m_tmKernel(nullptr)
{
    ContainerSpecification in_image("in_binary", ContainerSpecification::READ_ONLY);
    ContainerSpecification out_numbers("out_numbers", ContainerSpecification::REFERENCE);
    ContainerSpecification out_rot("out_rotation", ContainerSpecification::REFERENCE);


    ContainerSpecification out_deb("out_debug", ContainerSpecification::REFERENCE);

    m_argumentsSpecification.push_back(in_image);
    m_argumentsSpecification.push_back(out_numbers);
    m_argumentsSpecification.push_back(out_rot);

    m_argumentsSpecification.push_back(out_deb);

    m_settings.add(Option("templateDiscardFactor", OptionValue<double>(0.5, 0.5, 0.0, 1.0)));
    m_settings.add(Option("ratioDiscardFactor", OptionValue<double>(1.0, 1.0, 0.0, 1.0)));
    m_settings.add(Option("cellMargin", OptionValue<double>(0.05, 0.05, 0.0, 0.4)));
}

TemplateMatch::~TemplateMatch()
{
    ReleaseResources();
}

std::vector<Algorithm::ImplementationType> TemplateMatch::supportedImplementations() const
{
    std::vector<Algorithm::ImplementationType> v;
    v.push_back(CPU);
    v.push_back(GPU);
    v.push_back(OPENCV_GPU);
    return v;
}

bool TemplateMatch::exec()
{
    if (!m_implSet)
    {
        setImplementation(ImplementationType::OPENCV_GPU);
    }

    if (m_ocr.digits().size() == 0)
    {
        throw std::runtime_error("No ocr loaded");
    }

    auto in = m_arguments[0];

    std::vector<int> rotations;

    for (size_t i = 0; i < in->size(); ++i)
    {
        cv::Mat inImage = *in->get(i);
        auto outNumbers = std::make_shared<cv::Mat>();

        auto outDebug = std::make_shared<cv::Mat>();
        Rotation rot = Rotation::Zero;
        if (!inImage.empty())
        {
            if (m_activeImpl == CPU)
            {
                matchCPU(inImage, *outNumbers, rot, *outDebug);
                m_arguments[3]->add(outDebug);
            }
            else if (m_activeImpl == GPU)
            {
                matchGPU(inImage, *outNumbers, rot, *outDebug);
                m_arguments[3]->add(outDebug);
            }
            else if (m_activeImpl == OPENCV_GPU)
            {
                matchOCVGPU(inImage, *outNumbers, rot);
            }
            else
            {
                throw std::runtime_error("Invalid implementation selected");
            }
        }

        m_arguments[1]->add(outNumbers);
        rotations.push_back(static_cast<int>(rot));
    }

    cv::Mat(rotations).copyTo(*m_arguments[2]->get());

    return true;
}

void cellImage(cv::Mat& in, cv::Mat& out, double dist, double size, double margin)
{
    int nSize = size;
    out = cv::Mat(nSize*9, nSize*9, in.type());
    //std::cout << "(" << in.cols << "x" << in.rows << ") -> (" << nSize*9 << "x" << nSize*9 << ")" << std::endl;
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            int xS = j*dist + margin;
            int yS = i*dist + margin;
            int xT = j*nSize;
            int yT = i*nSize;
            in.rowRange(yS, yS+nSize)
              .colRange(xS, xS+nSize)
            .copyTo(out.rowRange(yT, yT+nSize)
                       .colRange(xT, xT+nSize));
        }
    }
}
