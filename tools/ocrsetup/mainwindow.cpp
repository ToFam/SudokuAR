#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <imageview.h>
#include <qimage_util.h>

#include <QHBoxLayout>
#include <QFileDialog>

using namespace std;
using namespace cv;

DevelGUIWindow::DevelGUIWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);


    m_view = new ImageView(this);
    QVBoxLayout* mainLayout = static_cast<QVBoxLayout*>(ui->centralWidget->layout());
    mainLayout->removeWidget(ui->view);
    delete ui->view;
    ui->view = m_view;
    mainLayout->insertWidget(0, m_view);


    connect(m_view, &ImageView::selected, this, &DevelGUIWindow::selected);

    connect(ui->pushButton, SIGNAL(pressed()), this, SLOT(added()));

    connect(ui->pbDetect, SIGNAL(pressed()), this, SLOT(detect()));

    connect(ui->pbFilePath, &QPushButton::clicked, this, [&]() {
        QString path = QFileDialog::getOpenFileName(this, tr("Open ORC input image"));

        if (!path.isEmpty())
        {
            ui->leFile->setText(path);
        }
    });
    connect(ui->pbOpen, &QPushButton::clicked, this, [&](){load(ui->leFile->text());});

    ui->digitView->setScene(new QGraphicsScene());




}

DevelGUIWindow::~DevelGUIWindow()
{
    delete ui;
}

void DevelGUIWindow::load(QString path)
{
    ui->leFile->setText(path);
    m_digits.clear();
    ui->listDigits->clear();
    ui->digitView->scene()->clear();

    m_orig = imread(path.toStdString(), IMREAD_COLOR);

    cvtColor( m_orig, m_img, COLOR_RGB2GRAY );
    GaussianBlur(m_img, m_img, Size(3,3), 0);
    threshold(m_img, m_img, 80, 255, cv::THRESH_BINARY_INV);

    m_view->setImage(m_img);
}

void DevelGUIWindow::selected(QRectF r)
{
    m_img.colRange((int)r.x(), (int)r.width() + (int)r.x()).rowRange((int)r.y(), (int)r.height() + (int)r.y()).copyTo(m_currentTemplate);
    ui->digitView->scene()->clear();
    ui->digitView->scene()->addPixmap(cvMatToQPixmap(m_currentTemplate));
    ui->digitView->scene()->setSceneRect(ui->digitView->scene()->itemsBoundingRect());
    ui->digitView->fitInView(ui->digitView->scene()->sceneRect(), Qt::KeepAspectRatio);
    m_currentRect = r;

    double scoreThresh = ui->sbThresh->value();

    vector<double> scores;
    double maxScore = 0.f;
    int maxScoreDigit = -1;
    for (Digit d : m_digits)
    {
        Mat res;

        if (m_currentTemplate.rows < d.templ.rows ||
            m_currentTemplate.cols < d.templ.cols )
        {
            maxScoreDigit = -1;
            break;
        }

        matchTemplate(m_currentTemplate, d.templ, res, cv::TM_CCOEFF);
        double max;
        minMaxLoc(res, nullptr, &max, nullptr, nullptr);

        scores.push_back(max);
        if (max > maxScore)
        {
            maxScore = max;
            maxScoreDigit = d.value;
        }
    }

    if (maxScoreDigit > -1 && maxScore > scoreThresh)
    {
        ui->lDetected->setText(QString::number(maxScoreDigit));

        ui->listDigits->clear();

        for (int i = 0; i < m_digits.size(); i++)
        {
            new QListWidgetItem(QString::number(m_digits[i].value) + " - " + QString::number(scores[i]), ui->listDigits);
        }
    }
    else
    {
        ui->lDetected->setText("-");

        ui->listDigits->clear();
        for (int i = 0; i < m_digits.size(); i++)
        {
            new QListWidgetItem(QString::number(m_digits[i].value), ui->listDigits);
        }
    }
}

void DevelGUIWindow::added()
{
    Digit d;
    m_currentTemplate.copyTo(d.templ);
    d.value = ui->leValue->text().toInt();
    m_digits.push_back(d);

    new QListWidgetItem(ui->leValue->text(), ui->listDigits);
}

void DevelGUIWindow::detect()
{
    Mat out;
    cvtColor( m_img, out, COLOR_GRAY2BGR );

    double dist = m_currentTemplate.rows / 9.0;
    double size = dist * 0.9;
    double margin = dist * 0.05;

    m_cellResolution = dist;
    double scoreThresh = ui->sbThresh->value();

    for (int row = 0; row < 9; row++)
    {
        for (int col = 0; col < 9; col++)
        {
            int left = m_currentRect.x() + col * dist + margin;
            int top =  m_currentRect.y() + row * dist + margin;
            int right = left + size;
            int bottom = top + size;

            Point2i p1(left, top);
            Point2i p2(right, bottom);
            cv::rectangle(out, p1, p2, Scalar(255, 0, 0));

            Mat target;
            m_img.rowRange(top, bottom).colRange(left, right).copyTo(target);

            vector<double> scores;
            double maxScore = 0.f;
            int maxScoreDigit = -1;
            for (Digit d : m_digits)
            {
                Mat res;

                if (target.rows < d.templ.rows ||
                    target.cols < d.templ.cols )
                {
                    maxScoreDigit = -1;
                    break;
                }

                matchTemplate(target, d.templ, res, cv::TM_CCOEFF);
                double max;
                minMaxLoc(res, nullptr, &max, nullptr, nullptr);

                scores.push_back(max);
                if (max > maxScore)
                {
                    maxScore = max;
                    maxScoreDigit = d.value;
                }
            }

            if (maxScoreDigit > -1 && maxScore > scoreThresh)
            {
                for (Digit d : m_digits)
                {
                    if (d.value == maxScoreDigit)
                    {
                        Mat detected;
                        Size s(target.cols, target.rows);
                        cv::resize(d.templ, detected, s);

                        Mat bla;
                        addWeighted(detected, 0.5, target, 0.5, 0, bla);
                        cvtColor(bla, out.rowRange(top, bottom).colRange(left, right), cv::COLOR_GRAY2BGR);

                        break;
                    }
                }
            }
        }
    }


    m_view->setImage(out);
}

void DevelGUIWindow::on_pushButton_2_clicked()
{
    QString path = QFileDialog::getSaveFileName(this, tr("Choose output save location"), QDir::currentPath() + "/ocr.yml", tr("Data file (*.yml *.yaml *.json *.xml)"));
    if (path.isEmpty())
        return;

    FileStorage fs(path.toStdString(), FileStorage::WRITE);

    fs << "cellResolution" << m_cellResolution;

    fs << "scoreThreshold" << ui->sbThresh->value();

    fs << "digits" << "[";

    for (Digit d : m_digits)
    {
        fs << "{";
        fs << "value" << d.value;
        fs << "matrix" << d.templ;
        fs <<  "}";
    }

    fs << "]";
}
