#include "ContainerList.h"

#include <QInputDialog>

ContainerList::ContainerList() : QAbstractListModel()
{
    m_list.push_back("Frame");
    m_view = nullptr;
}

QStringList ContainerList::list() const
{
    return m_list;
}

void ContainerList::setView(QListView *view)
{
    m_view = view;
}

void ContainerList::clear()
{
    m_list.clear();
}

int ContainerList::indexOf(QString label)
{
    return m_list.indexOf(label);
}

int ContainerList::rowCount(const QModelIndex &parent) const
{
    return m_list.size();
}

QVariant ContainerList::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole)
    {
        return m_list[index.row()];
    }

    return QVariant();
}

void ContainerList::add()
{
    QWidget* parent = nullptr;
    if (m_view != nullptr)
        parent = m_view;

    QString text = QInputDialog::getText(parent, tr("Container label"),
                                         tr("Enter a label for the new container:"));
    if (!text.isEmpty())
    {
        add(text);
    }
}

void ContainerList::add(QString s)
{
    beginInsertRows(QModelIndex(), m_list.size(), m_list.size());
    m_list.append(s);
    endInsertRows();
}

void ContainerList::remove()
{
    if (m_view != nullptr)
    {
        auto sl = m_view->selectionModel()->selectedRows();
        if (sl.size() > 0)
        {
            QModelIndex& i = sl.first();

            if (m_list[i.row()] != "Frame")
            {
                beginRemoveRows(QModelIndex(), i.row(), i.row());
                m_list.removeAt(i.row());
                endRemoveRows();
            }
        }
    }
}

void ContainerList::moveCurrentUp()
{
    if (m_view != nullptr)
    {
        auto sl = m_view->selectionModel()->selectedRows();
        if (sl.size() > 0)
        {
            QModelIndex& modelIndex = sl.first();

            int index = modelIndex.row();
            QString str = m_list[index];

            if (index > 0 && m_list[index] != "Frame")
            {
                beginRemoveRows(QModelIndex(), index, index);
                m_list.removeAt(index);
                endRemoveRows();

                index--;
                beginInsertRows(QModelIndex(), index, index);
                m_list.insert(index, str);
                endInsertRows();

                QSignalBlocker b1(m_view->selectionModel());
                m_view->selectionModel()->select(this->index(index), QItemSelectionModel::ClearAndSelect);
            }
        }
    }
}

void ContainerList::moveCurrentDown()
{
    if (m_view != nullptr)
    {
        auto sl = m_view->selectionModel()->selectedRows();
        if (sl.size() > 0)
        {
            QModelIndex& modelIndex = sl.first();

            int index = modelIndex.row();
            QString str = m_list[index];

            if (index < m_list.size() - 1 && m_list[index] != "Frame")
            {
                beginRemoveRows(QModelIndex(), index, index);
                m_list.removeAt(index);
                endRemoveRows();

                index++;
                beginInsertRows(QModelIndex(), index, index);
                m_list.insert(index, str);
                endInsertRows();

                QSignalBlocker b1(m_view->selectionModel());
                m_view->selectionModel()->select(this->index(index), QItemSelectionModel::ClearAndSelect);
            }
        }
    }
}
