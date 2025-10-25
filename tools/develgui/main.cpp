#include "DevelGUIWindow.h"

#include <QApplication>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include <QString>
#include <QFile>
#include <QTextStream>

void parseSudokuFile()
{
    QFile f("/home/tofam/Downloads/hexa.txt");
    f.open(QFile::ReadOnly);

    QTextStream s(&f);

    std::vector<std::vector<int>> numbers;
    int n = -1;
    while(true)
    {
        QString line = s.readLine();
        if (line.isEmpty())
        {
            if (s.atEnd())
                break;
            else {
                continue;
            }
        }

        QStringList l = line.split(' ');

        std::vector<int> lineNums;
        for (QString p : l)
        {
            if (p == "*")
                lineNums.push_back(-1);
            else {
                bool ok;
                int num = p.toInt(&ok, 16);
                if (ok)
                    lineNums.push_back(num);
            }
        }

        if (lineNums.size() > 0 && n == -1)
        {
            n = lineNums.size();
        }
        else if (lineNums.size() > 0 && lineNums.size() != n) {
            return;
        }

        numbers.push_back(lineNums);
    }

    cv::Mat_<int> out(n * n, 1);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            out.at<int>(i * n + j, 0) = numbers[i][j] + 1;
        }
    }

    cv::FileStorage fs("/data/sudoku/nexport.yml", cv::FileStorage::WRITE);
    fs << "field" << out;
}

int main(int argc, char** argv)
{
    //parseSudokuFile();
    //return 0;

    cv::ocl::setUseOpenCL(true);

    std::cout << cv::ocl::Device::getDefault().name() << std::endl;

    QApplication a(argc, argv);

    DevelGUIWindow w;
    w.show();

    return a.exec();
}
