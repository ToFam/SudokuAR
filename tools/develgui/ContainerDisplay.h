#pragma once

#include <memory>

#include <QWidget>

#include <Container.h>
#include <imageview.h>

class QFrame;
class QLabel;
class QToolButton;

class ContainerDisplay : public QWidget
{
    Q_OBJECT

signals:
    void selected(const QRectF&);

public:
    ContainerDisplay(QWidget* parent = nullptr);

    void display(std::shared_ptr<Container> container);

    void setControlls(QFrame* frame, QLabel* label, QToolButton* left, QToolButton* right);

    cv::Mat currentImage() const;

    void setIndex(int index);

private slots:
    void left();
    void right();

private:
    ImageView* m_view;

    QFrame* m_containerDisplay;
    QLabel* m_containerDisplayLabel;
    QToolButton* m_containerDisplayLeft;
    QToolButton* m_containerDisplayRight;

    std::shared_ptr<Container> m_container;
    int m_index;
};
