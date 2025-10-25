#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <vector>

#include <opencv2/opencv.hpp>

namespace Ui {
class MainWindow;
}

class ImageView;

class DevelGUIWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit DevelGUIWindow(QWidget *parent = nullptr);
    ~DevelGUIWindow();

    void load(QString path);

private slots:
    void selected(QRectF r);
    void added();
    void detect();

    void on_pushButton_2_clicked();

private:
    Ui::MainWindow *ui;

    cv::Mat m_img;
    cv::Mat m_orig;

    cv::Mat m_currentTemplate;
    QRectF m_currentRect;

    struct Digit
    {
        cv::Mat templ;
        int value;
    };

    std::vector<Digit> m_digits;

    double m_cellResolution;

    ImageView* m_view;
};

#endif // MAINWINDOW_H
