#include "AlgorithmSettings.h"

AlgorithmSettings::AlgorithmSettings()
{

}

void AlgorithmSettings::add(Option opt)
{
    m_settings[opt.name()] = opt;
}

Option& AlgorithmSettings::get(std::string name)
{
    return m_settings[name];
}

std::vector<Option*> AlgorithmSettings::getOptions()
{
    std::vector<Option*> rtn;
    for (auto it = m_settings.begin(); it != m_settings.end(); ++it)
    {
        rtn.push_back(&it->second);
    }
    return rtn;
}

// =========================================================

Option::Option() : m_set(false)
{

}

Option::Option(std::string name) : m_set(false), m_name(name)
{

}

Option::Option(std::string name, OptionValue<int> intValue) : m_name(name)
{
    setInt(intValue);
}

Option::Option(std::string name, OptionValue<float> floatValue) : m_name(name)
{
    setFloat(floatValue);
}

Option::Option(std::string name, OptionValue<bool> boolValue) : m_name(name)
{
    setBool(boolValue);
}

Option::Option(std::string name, OptionValue<double> doubleValue) : m_name(name)
{
    setDouble(doubleValue);
}

bool Option::valid() const
{
    return m_name.size() != 0;
}

bool Option::isSet() const
{
    return m_set;
}

Option::Type Option::type() const
{
    if (!m_set) throw Not_Set();
    return m_type;
}

std::string Option::name() const
{
    return m_name;
}

void Option::setName(const std::string &name)
{
    m_name = name;
}

std::string Option::description() const
{
    return m_description;
}

void Option::setDescription(const std::string &description)
{
    m_description = description;
}

void Option::setInt(OptionValue<int> v)
{
    m_set = true;
    m_type = INT;
    m_int = v;
}

void Option::setBool(OptionValue<bool> v)
{
    m_set = true;
    m_type = BOOL;
    m_bool = v;
}

void Option::setFloat(OptionValue<float> v)
{
    m_set = true;
    m_type = FLOAT;
    m_float = v;
}

void Option::setDouble(OptionValue<double> v)
{
    m_set = true;
    m_type = DOUBLE;
    m_double = v;
}

void Option::setIntValue(int i)
{
    if (!m_set) throw Not_Set();
    if (m_type != INT) throw Bad_Type();
    m_int.setValue(i);
}

void Option::setBoolValue(bool b)
{
    if (!m_set) throw Not_Set();
    if (m_type != BOOL) throw Bad_Type();
    m_bool.setValue(b);
}

void Option::setFloatValue(float f)
{
    if (!m_set) throw Not_Set();
    if (m_type != FLOAT) throw Bad_Type();
    m_float.setValue(f);
}

void Option::setDoubleValue(double d)
{
    if (!m_set) throw Not_Set();
    if (m_type != DOUBLE) throw Bad_Type();
    m_double.setValue(d);
}

OptionValue<int> Option::valueInt() const
{
    if (!m_set) throw Not_Set();
    if (m_type != INT) throw Bad_Type();
    return m_int;
}

OptionValue<bool> Option::valueBool() const
{
    if (!m_set) throw Not_Set();
    if (m_type != BOOL) throw Bad_Type();
    return m_bool;
}

OptionValue<float> Option::valueFloat() const
{
    if (!m_set) throw Not_Set();
    if (m_type != FLOAT) throw Bad_Type();
    return m_float;
}

OptionValue<double> Option::valueDouble() const
{
    if (!m_set) throw Not_Set();
    if (m_type != DOUBLE) throw Bad_Type();
    return m_double;
}
