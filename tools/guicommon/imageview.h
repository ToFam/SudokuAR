#ifndef IMAGEVIEW_H
#define IMAGEVIEW_H

#include <QGraphicsView>
#include <QWheelEvent>
#include <QGraphicsItem>

#include <opencv2/opencv.hpp>

#include <vector>

class ImageView : public QGraphicsView
{
    Q_OBJECT

signals:
    void selected(const QRectF& area);

public:
    ImageView(QWidget* parent = nullptr);

public slots:
    void setImage(cv::Mat image);

    void addLine(QColor color, QPointF from, QPointF to);

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

    void keyPressEvent(QKeyEvent* event) override;

private:
    void addSelectionRect();

    bool m_dragging;
    bool m_selecting;
    QPoint m_dragStart;

    QGraphicsRectItem* m_selectionRect;
};

#endif // IMAGEVIEW_H
