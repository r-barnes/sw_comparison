# include (clark_support.pri)
include (../../ugene_version.pri)
include (../../ugene_globals.pri)

PLUGIN_ID=clark_support
PLUGIN_NAME=CLARK external tool support
PLUGIN_VENDOR=Unipro
PLUGIN_DEPENDS=ngs_reads_classification$${D}:$${UGENE_VERSION};external_tool_support$${D}:$${UGENE_VERSION}

include( ../../ugene_plugin_common.pri )

LIBS += -L../../$$out_dir()/plugins
LIBS += -lngs_reads_classification$$D

