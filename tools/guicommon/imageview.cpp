#include "imageview.h"
#include "qimage_util.h"

#include <QScrollBar>

ImageView::ImageView(QWidget* parent)
    : QGraphicsView(parent)
{
    setDragMode(NoDrag);
    setMouseTracking(true);
    setScene(new QGraphicsScene(this));
    m_dragging = false;
    m_selecting = false;

    addSelectionRect();
}

void ImageView::wheelEvent(QWheelEvent *event)
{
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    double scaleFactor = 1.15;
    if(event->delta() > 0) {
        scale(scaleFactor, scaleFactor);
    } else {
        scale(1.0 / scaleFactor, 1.0 / scaleFactor);
    }
}

void ImageView::addSelectionRect()
{
    m_selectionRect = new QGraphicsRectItem();
    m_selectionRect->setPen(QPen(QColor(0, 255, 255)));
    m_selectionRect->setBrush(QBrush(QColor(0, 80, 100, 100)));
    m_selectionRect->setRect(0,0,100,100);
    m_selectionRect->hide();
    scene()->addItem(m_selectionRect);
}

void ImageView::setImage(cv::Mat image)
{
    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(cvMatToQPixmap(image));

    scene()->clear();

    scene()->addItem(item);

    addSelectionRect();
}

void ImageView::addLine(QColor color, QPointF from, QPointF to)
{
    QLineF line(from, to);
    QGraphicsLineItem* lineItem = new QGraphicsLineItem(line);
    lineItem->setPen(QPen(color));
    scene()->addItem(lineItem);
}

void ImageView::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton)
    {
        m_dragging = true;
        m_dragStart = event->pos();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
    }
    if (event->button() == Qt::LeftButton && !m_dragging)
    {
        m_selecting = true;
        QPointF p = mapToScene(event->pos());
        if (p.x() < 0) p.setX(0);
        if (p.y() < 0) p.setY(0);
        if (p.x() > scene()->width()) p.setX(scene()->width());
        if (p.y() > scene()->height()) p.setY(scene()->height());

        m_selectionRect->setRect((int)p.x(), (int)p.y(), 0, 0);
        m_selectionRect->show();
    }
    else
    {
        QGraphicsView::mousePressEvent(event);
    }
}

void ImageView::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton)
    {
        m_dragging = false;
        setCursor(Qt::ArrowCursor);
        event->accept();
    }
    if (event->button() == Qt::LeftButton)
    {
        m_selecting = false;

        QRectF r = m_selectionRect->rect();
        if (r.width() + r.height() > 0.f)
        {
            emit selected(m_selectionRect->rect());
        }
        else
        {
            m_selectionRect->hide();
        }
        event->accept();
    }
    else
    {
        QGraphicsView::mouseReleaseEvent(event);
    }
}

void ImageView::mouseMoveEvent(QMouseEvent *event)
{
    if (m_dragging)
    {
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - (event->x() - m_dragStart.x()));
        verticalScrollBar()->setValue(verticalScrollBar()->value() - (event->y() - m_dragStart.y()));
        m_dragStart = event->pos();
        event->accept();
    }
    if (m_selecting)
    {
        QRectF r = m_selectionRect->rect();
        QPointF p = mapToScene(event->pos());
        if (p.x() < 0) p.setX(0);
        if (p.y() < 0) p.setY(0);
        if (p.x() > scene()->width()) p.setX(scene()->width());
        if (p.y() > scene()->height()) p.setY(scene()->height());
        m_selectionRect->setRect((int)r.x(), (int)r.y(), (int)p.x() - (int)r.x(), (int)p.y() - (int)r.y());
        event->accept();
    }
    else
    {
        QGraphicsView::mouseMoveEvent(event);
    }
}

void ImageView::keyPressEvent(QKeyEvent* event)
{

}
