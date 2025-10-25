#pragma once

#include <memory>

#include <Algorithm.h>
#include <AlgorithmSettings.h>
#include <Container.h>

#include "ContainerList.h"

#include <QObject>

#include <opencv2/core/persistence.hpp>

class QWidget;

class GUIAlgoElement : public QObject
{
    Q_OBJECT

signals:
    void remove();
    void up();
    void down();
    void liveUpdate();

public:
    GUIAlgoElement(QString name, std::unique_ptr<Algorithm> algo, ContainerList& containerList);
    virtual ~GUIAlgoElement();

    QString name() const { return m_name; }
    Algorithm* algo() const { return m_algo.get(); }
    QWidget* widget() const { return m_parameterBox; }

    void prepare(std::map<std::string, std::shared_ptr<Container>>& containerStack);

    void loadFromFile(cv::FileNode& node);
    void saveToFile(cv::FileStorage& fs);

private:
    ContainerList& m_containerList;

    QString m_name;
    std::unique_ptr<Algorithm> m_algo;
    QWidget* m_parameterBox;

    std::map<std::string, QWidget*> m_argumentWidgets;
    std::map<std::string, QWidget*> m_optionWidgets;
};
