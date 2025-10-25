#include "ContainerDisplay.h"

#include <QVBoxLayout>

#include <QLabel>
#include <QFrame>
#include <QToolButton>

ContainerDisplay::ContainerDisplay(QWidget* parent) : QWidget(parent)
{
    m_view = new ImageView(this);
    setLayout(new QVBoxLayout());
    layout()->addWidget(m_view);
    connect(m_view, &ImageView::selected, this, &ContainerDisplay::selected);
}

void ContainerDisplay::display(std::shared_ptr<Container> container)
{
    m_container = container;

    m_index = 0;
    m_view->setImage(*m_container->get());

    int size = m_container->size();
    m_containerDisplay->setEnabled(size > 1);
    m_containerDisplayLabel->setText(QString("1 / %1").arg(size));
}

void ContainerDisplay::setControlls(QFrame *frame, QLabel *label, QToolButton *left, QToolButton *right)
{
    m_containerDisplay = frame;
    m_containerDisplayLabel = label;
    m_containerDisplayLeft = left;
    m_containerDisplayRight = right;

    connect(m_containerDisplayLeft, &QToolButton::clicked, this, &ContainerDisplay::left);
    connect(m_containerDisplayRight, &QToolButton::clicked, this, &ContainerDisplay::right);
}

cv::Mat ContainerDisplay::currentImage() const
{
    return *m_container->get(m_index);
}

void ContainerDisplay::left()
{
    if (m_index > 0)
    {
        setIndex(m_index - 1);
    }
}

void ContainerDisplay::right()
{
    if (m_index < m_container->size() - 1)
    {
        setIndex(m_index + 1);
    }
}

void ContainerDisplay::setIndex(int index)
{
    m_index = index;

    m_view->setImage(*m_container->get(m_index));

    m_containerDisplayLabel->setText(QString("%1 / %2").arg(m_index + 1).arg(m_container->size()));
}
