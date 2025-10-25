#pragma once

#include <map>
#include <vector>
#include <string>

template <typename T>
class OptionValue
{
public:
    OptionValue();
    OptionValue(T def);
    OptionValue(T value, T def);
    OptionValue(T value, T def, T min, T max);

    T defaultValue() const;
    void setDefaultValue(const T &defaultValue);

    T value() const;
    void setValue(const T &value);

    T min() const;
    void setMin(const T &min);

    T max() const;
    void setMax(const T &max);

private:
    T            m_default;
    T            m_value;
    T            m_min;
    T            m_max;
};

class Option
{
public:
    enum Type
    {
        BOOL,
        INT,
        FLOAT,
        DOUBLE
    };

    Type type() const;

    struct Bad_Type {};
    struct Not_Set {};

private:
    union
    {
        OptionValue<int> m_int;
        OptionValue<bool> m_bool;
        OptionValue<float> m_float;
        OptionValue<double> m_double;
    };

public:
    Option();
    Option(std::string name);
    Option(std::string name, OptionValue<int> intValue);
    Option(std::string name, OptionValue<bool> boolValue);
    Option(std::string name, OptionValue<float> floatValue);
    Option(std::string name, OptionValue<double> doubleValue);

    bool valid() const;
    bool isSet() const;

    void setInt(OptionValue<int> v);
    void setBool(OptionValue<bool> v);
    void setFloat(OptionValue<float> v);
    void setDouble(OptionValue<double> v);

    void setIntValue(int i);
    void setBoolValue(bool b);
    void setFloatValue(float f);
    void setDoubleValue(double d);

    OptionValue<int> valueInt() const;
    OptionValue<bool> valueBool() const;
    OptionValue<float> valueFloat() const;
    OptionValue<double> valueDouble() const;

    std::string name() const;
    void setName(const std::string &name);

    std::string description() const;
    void setDescription(const std::string &description);

private:
    Type        m_type;
    bool        m_set;
    std::string m_name;
    std::string m_description;
};

class AlgorithmSettings
{
public:
    AlgorithmSettings();

    void add(Option opt);
    Option& get(std::string name);

    std::vector<Option*> getOptions();

private:
    std::map<std::string, Option> m_settings;
};

// =========================================================

template<typename T>
OptionValue<T>::OptionValue()
{

}

template<typename T>
OptionValue<T>::OptionValue(T def) : m_default(def)
{

}

template<typename T>
OptionValue<T>::OptionValue(T value, T def) : m_default(def), m_value(value)
{

}

template<typename T>
OptionValue<T>::OptionValue(T value, T def, T min, T max)
 : m_default(def), m_value(value), m_min(min), m_max(max)
{

}

template<typename T>
T OptionValue<T>::value() const
{
    return m_value;
}

template<typename T>
void OptionValue<T>::setValue(const T &value)
{
    m_value = value;
}

template<typename T>
T OptionValue<T>::min() const
{
    return m_min;
}

template<typename T>
void OptionValue<T>::setMin(const T &min)
{
    m_min = min;
}

template<typename T>
T OptionValue<T>::max() const
{
    return m_max;
}

template<typename T>
void OptionValue<T>::setMax(const T &max)
{
    m_max = max;
}

template<typename T>
T OptionValue<T>::defaultValue() const
{
    return m_default;
}

template<typename T>
void OptionValue<T>::setDefaultValue(const T &defaultValue)
{
    m_default = defaultValue;
}
