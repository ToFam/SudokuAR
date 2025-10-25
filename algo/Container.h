#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>

class ContainerSpecification
{
public:
    enum Type
    {
        READ_ONLY,
        COPY,
        REFERENCE
    };

    enum DataType
    {
        GENERIC,
        IMAGE,
        LIST
    };

    ContainerSpecification();
    ContainerSpecification(std::string name, Type type, DataType dataType = GENERIC);

    std::string name() const;
    void setName(const std::string &name);

    Type type() const;
    void setType(const Type &type);

    DataType dataType() const;
    void setDataType(const DataType &dataType);

private:
    std::string m_name;
    DataType m_dataType;
    Type m_type;
};

class Container
{
public:
    Container();
    Container(ContainerSpecification spec);

    std::shared_ptr<cv::Mat> get() const;
    void set(std::shared_ptr<cv::Mat> mat);

    std::shared_ptr<cv::Mat> get(size_t index) const;
    void add(std::shared_ptr<cv::Mat> item);
    size_t size() const;

    ContainerSpecification spec() const;

private:
    ContainerSpecification m_spec;
    std::vector<std::shared_ptr<cv::Mat>> m_list;
};
