# include (umuscle.pri)

PLUGIN_ID=umuscle
PLUGIN_NAME=Muscle3
PLUGIN_VENDOR=Unipro
CONFIG += warn_off

include( ../../ugene_plugin_common.pri )

LIBS += -lqscore$$D

