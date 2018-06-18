#ifndef __CONFIG_FILE_H__
#define __CONFIG_FILE_H__

#include <string>
#include <map>

#include <qstring.h>

class ConfigFile {

public:
	ConfigFile(const QString & configFile);

	bool isGood() const;
	QString Value(const QString & section, const QString & entry, double value);
	QString Value(const QString & section, const QString & entry, const QString & value) const;

private:
	QString Value( const QString & section, const QString & entry) const;
	std::string trim(const std::string  & source, char const* delims = " \t\r\n");

	std::map<QString, QString> content_;
	bool good;
};

#endif

