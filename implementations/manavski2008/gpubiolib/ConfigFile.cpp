#include "ConfigFile.h"

#include <fstream>

#define QT_NO_CAST_ASCII 

ConfigFile::ConfigFile(const QString & configFile) {


	QByteArray asciiName = configFile.toAscii();

	std::ifstream file(asciiName.constData());
	good = file.good();

	std::string line;
	std::string name;
	std::string value;
	std::string inSection;
	int posEqual;
	while (std::getline(file,line)) {

		if (! line.length()) continue;

		if (line[0] == '#') continue;
		if (line[0] == ';') continue;

		if (line[0] == '[') {
			inSection=trim(line.substr(1,line.find(']')-1));
			continue;
		}

		posEqual=line.find('=');
		name  = trim(line.substr(0,posEqual));
		value = trim(line.substr(posEqual+1));

		std::string skey = inSection+'/'+name;
		QString qkey = skey.c_str();
		content_[qkey]=value.c_str();
	}
}


bool ConfigFile::isGood( ) const
{
	return good;
}

QString ConfigFile::Value(const QString & section, const QString & entry, double value) {
	try {
		return Value(section, entry);
	} catch(const char *) {
		QString tmpStr = section+'/'+entry;
		return QString::number(value);
	}
}

QString ConfigFile::Value(const QString & section, const QString & entry, const QString & value) const
{
	try {
		return Value(section, entry);
	} catch(const char *) {
		return value;
	}
}



//---private---------------------------------------------------------------------------------------

QString ConfigFile::Value(const QString & section, const QString & entry) const {

  std::map<QString,QString>::const_iterator ci = content_.find(section + '/' + entry);

  if (ci == content_.end()) throw "does not exist";

  return ci->second;
}

std::string ConfigFile::trim(const std::string  & source, char const* delims) {
std::string result(source);
  std::string::size_type index = result.find_last_not_of(delims);
  if(index != std::string::npos)
    result.erase(++index);

  index = result.find_first_not_of(delims);
  if(index != std::string::npos)
    result.erase(0, index);
  else
    result.erase();

  return result;
}

