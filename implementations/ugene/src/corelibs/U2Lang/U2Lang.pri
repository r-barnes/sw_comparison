# include (U2Lang.pri)

MODULE_ID=U2Lang
include( ../../ugene_lib_common.pri )

QT += xml widgets
DEFINES+= QT_FATAL_ASSERT BUILDING_U2LANG_DLL

LIBS += -L../../$$out_dir()
LIBS += -lU2Core$$D
DESTDIR = ../../$$out_dir()

unix {
    target.path = $$UGENE_INSTALL_DIR/
    INSTALLS += target
}
