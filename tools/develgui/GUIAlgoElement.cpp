#include "GUIAlgoElement.h"

#include <QWidget>
#include <QLabel>
#include <QComboBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QToolButton>

#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>

GUIAlgoElement::GUIAlgoElement(QString name, std::unique_ptr<Algorithm> algo, ContainerList& containerList) : m_containerList(containerList), m_name(name), m_algo(std::move(algo))
{
    m_parameterBox = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout;

    if (m_algo->supportedImplementations().size() > 1)
    {
        QHBoxLayout* l = new QHBoxLayout;

        QComboBox* b = new QComboBox;
        for (Algorithm::ImplementationType type : m_algo->supportedImplementations())
        {
            switch(type)
            {
            case Algorithm::ImplementationType::CPU:
                b->addItem("CPU");
                break;
            case Algorithm::ImplementationType::GPU:
                b->addItem("GPU");
                break;
            case Algorithm::ImplementationType::OPENCV_CPU:
                b->addItem("OCV_CPU");
                break;
            case Algorithm::ImplementationType::OPENCV_GPU:
                b->addItem("OCV_GPU");
                break;
            }
        }

        connect(b, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
                this, [&](int index){ m_algo->setImplementation(m_algo->supportedImplementations()[index]); });
        l->addWidget(new QLabel(tr("Implementation")));
        l->addWidget(b);
        layout->addItem(l);
    }

    for (const auto& spec : m_algo->specification())
    {
        QHBoxLayout* l = new QHBoxLayout();

        QComboBox* b = new QComboBox;
        b->setModel(&m_containerList);
        m_argumentWidgets[spec.name()] = b;

        l->addWidget(new QLabel(QString::fromStdString(spec.name())));
        l->addWidget(b);
        layout->addItem(l);
    }

    for (Option* opt : m_algo->settings().getOptions())
    {
        QHBoxLayout* l = new QHBoxLayout();

        bool label = true;
        QWidget* w = nullptr;
        QMetaObject::Connection con;
        switch (opt->type())
        {
        case Option::INT:
        {
            QSpinBox* b = new QSpinBox();
            b->setRange(opt->valueInt().min(), opt->valueInt().max());
            b->setValue(opt->valueInt().defaultValue());
            if (m_algo->canDoLiveUpdate())
                con = connect(b, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &GUIAlgoElement::liveUpdate);
            w = b;
            break;
        }
        case Option::BOOL:
        {
            label = false;
            QCheckBox* b = new QCheckBox(QString::fromStdString(opt->name()));
            b->setChecked(opt->valueBool().defaultValue());
            if (m_algo->canDoLiveUpdate())
                con = connect(b, &QCheckBox::toggled, this, &GUIAlgoElement::liveUpdate);
            w = b;
            break;
        }
        case Option::FLOAT:
        {
            QDoubleSpinBox* b = new QDoubleSpinBox();
            b->setRange(opt->valueFloat().min(), opt->valueFloat().max());
            b->setValue(opt->valueFloat().defaultValue());
            if (m_algo->canDoLiveUpdate())
                con = connect(b, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &GUIAlgoElement::liveUpdate);
            w = b;
            break;
        }
        case Option::DOUBLE:
        {
            QDoubleSpinBox* b = new QDoubleSpinBox();
            b->setRange(opt->valueDouble().min(), opt->valueDouble().max());
            b->setValue(opt->valueDouble().defaultValue());
            if (m_algo->canDoLiveUpdate())
                con = connect(b, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &GUIAlgoElement::liveUpdate);
            w = b;
            break;
        }
        }

        if (w)
        {
            m_optionWidgets[opt->name()] = w;

            if (label)
                l->addWidget(new QLabel(QString::fromStdString(opt->name())));

            l->addWidget(w);
            layout->addItem(l);
        }
    }

    QHBoxLayout* btnLayout = new QHBoxLayout;
    QPushButton* removeBtn = new QPushButton(QObject::tr("Remove", "GUIAlgoElement"));
    QToolButton* upBtn = new QToolButton();
    QToolButton* downBtn = new QToolButton();
    upBtn->setArrowType(Qt::UpArrow);
    downBtn->setArrowType(Qt::DownArrow);
    connect(removeBtn, &QPushButton::pressed, this, &GUIAlgoElement::remove);
    connect(upBtn, &QToolButton::pressed, this, &GUIAlgoElement::up);
    connect(downBtn, &QToolButton::pressed, this, &GUIAlgoElement::down);
    btnLayout->addWidget(upBtn);
    btnLayout->addWidget(downBtn);
    btnLayout->addStretch();
    btnLayout->addWidget(removeBtn);

    layout->addStretch();
    layout->addItem(btnLayout);

    m_parameterBox->setLayout(layout);
}

GUIAlgoElement::~GUIAlgoElement()
{

}

void GUIAlgoElement::prepare(std::map<std::string, std::shared_ptr<Container> > &containerStack)
{
    m_algo->clearContainerStack();

    for (const auto& spec : m_algo->specification())
    {
        QComboBox* box = static_cast<QComboBox*>(m_argumentWidgets[spec.name()]);

        auto it = containerStack.find(box->currentText().toStdString());
        if (it != containerStack.end())
        {
            m_algo->addContainer(it->second);
        }
    }

    for (Option* opt : m_algo->settings().getOptions())
    {
        switch (opt->type())
        {
        case Option::INT:
        {
            QSpinBox* b = static_cast<QSpinBox*>(m_optionWidgets[opt->name()]);
            opt->setIntValue(b->value());
            break;
        }
        case Option::BOOL:
        {
            QCheckBox* b = static_cast<QCheckBox*>(m_optionWidgets[opt->name()]);
            opt->setBoolValue(b->isChecked());
            break;
        }
        case Option::FLOAT:
        {
            QDoubleSpinBox* b = static_cast<QDoubleSpinBox*>(m_optionWidgets[opt->name()]);
            opt->setFloatValue(b->value());
            break;
        }
        case Option::DOUBLE:
        {
            QDoubleSpinBox* b = static_cast<QDoubleSpinBox*>(m_optionWidgets[opt->name()]);
            opt->setDoubleValue(b->value());
            break;
        }
        }
    }
}

void GUIAlgoElement::loadFromFile(cv::FileNode &node)
{
    for (const auto& spec : m_algo->specification())
    {
        QComboBox* box = static_cast<QComboBox*>(m_argumentWidgets[spec.name()]);
        cv::FileNode n = node[spec.name()];
        if (!n.isNone())
        {
            QString label = QString::fromStdString(static_cast<std::string>(n));
            int index = m_containerList.indexOf(label);
            box->setCurrentIndex(index);
        }
    }

    for (Option* opt : m_algo->settings().getOptions())
    {
        cv::FileNode n = node[opt->name()];
        if (!n.isNone())
        {
            switch (opt->type())
            {
            case Option::INT:
            {
                QSpinBox* b = static_cast<QSpinBox*>(m_optionWidgets[opt->name()]);
                QSignalBlocker block(b);
                b->setValue(static_cast<int>(n));
                break;
            }
            case Option::BOOL:
            {
                QCheckBox* b = static_cast<QCheckBox*>(m_optionWidgets[opt->name()]);
                QSignalBlocker block(b);
                b->setChecked(static_cast<int>(n));
                break;
            }
            case Option::FLOAT:
            {
                QDoubleSpinBox* b = static_cast<QDoubleSpinBox*>(m_optionWidgets[opt->name()]);
                QSignalBlocker block(b);
                b->setValue(static_cast<float>(n));
                break;
            }
            case Option::DOUBLE:
            {
                QDoubleSpinBox* b = static_cast<QDoubleSpinBox*>(m_optionWidgets[opt->name()]);
                QSignalBlocker block(b);
                b->setValue(static_cast<double>(n));
                break;
            }
            }
        }
    }
}

void GUIAlgoElement::saveToFile(cv::FileStorage &fs)
{
    for (const auto& spec : m_algo->specification())
    {
        QComboBox* box = static_cast<QComboBox*>(m_argumentWidgets[spec.name()]);
        fs << spec.name() << box->currentText().toStdString();
    }

    for (Option* opt : m_algo->settings().getOptions())
    {
        fs << opt->name();
        switch (opt->type())
        {
        case Option::INT:
        {
            QSpinBox* b = static_cast<QSpinBox*>(m_optionWidgets[opt->name()]);
            fs << b->value();
            break;
        }
        case Option::BOOL:
        {
            QCheckBox* b = static_cast<QCheckBox*>(m_optionWidgets[opt->name()]);
            fs << b->isChecked();
            break;
        }
        case Option::FLOAT:
        {
            QDoubleSpinBox* b = static_cast<QDoubleSpinBox*>(m_optionWidgets[opt->name()]);
            fs << b->value();
            break;
        }
        case Option::DOUBLE:
        {
            QDoubleSpinBox* b = static_cast<QDoubleSpinBox*>(m_optionWidgets[opt->name()]);
            fs << b->value();
            break;
        }
        }
    }
}
