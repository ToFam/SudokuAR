#include "Container.h"

ContainerSpecification::ContainerSpecification()
{

}

ContainerSpecification::ContainerSpecification(std::string name, ContainerSpecification::Type type, ContainerSpecification::DataType dataType)
 : m_name(name), m_dataType(dataType), m_type(type)
{

}

std::string ContainerSpecification::name() const
{
    return m_name;
}

void ContainerSpecification::setName(const std::string &name)
{
    m_name = name;
}

ContainerSpecification::Type ContainerSpecification::type() const
{
    return m_type;
}

void ContainerSpecification::setType(const Type &type)
{
    m_type = type;
}

ContainerSpecification::DataType ContainerSpecification::dataType() const
{
    return m_dataType;
}

void ContainerSpecification::setDataType(const DataType &dataType)
{
    m_dataType = dataType;
}

Container::Container()
{

}

Container::Container(ContainerSpecification spec)
    : m_spec(spec)
{

}

std::shared_ptr<cv::Mat> Container::get() const
{
    return m_list.front();
}

void Container::set(std::shared_ptr<cv::Mat> mat)
{
    if (m_list.size() == 0)
        m_list.emplace_back();

    if (m_spec.type() == ContainerSpecification::COPY)
    {
        m_list[0] = std::make_shared<cv::Mat>(mat->clone());
    }
    else
    {
        m_list[0] = mat;
    }
}

void Container::add(std::shared_ptr<cv::Mat> item)
{
    if (m_list.size() == 1 && m_list[0]->empty())
        m_list.clear();

    if (m_spec.type() == ContainerSpecification::COPY)
    {
        m_list.push_back(std::make_shared<cv::Mat>(item->clone()));
    }
    else
    {
        m_list.push_back(item);
    }
}

size_t Container::size() const
{
    return m_list.size();
}

ContainerSpecification Container::spec() const
{
    return m_spec;
}

std::shared_ptr<cv::Mat> Container::get(size_t index) const
{
    return m_list[index];
}
