#pragma once

#include <QAbstractListModel>
#include <QListView>

class ContainerList : public QAbstractListModel
{
    Q_OBJECT

public:
    ContainerList();

public:
    QStringList list() const;

public:
    int rowCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;

    void setView(QListView* view);

    void clear();

    int indexOf(QString label);

public slots:
    void add();
    void add(QString s);
    void remove();

    void moveCurrentUp();
    void moveCurrentDown();

private:
    QStringList m_list;

    QListView* m_view;
};
