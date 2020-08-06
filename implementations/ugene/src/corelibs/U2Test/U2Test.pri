# include (U2Test.pri)

MODULE_ID=U2Test
include( ../../ugene_lib_common.pri )

QT += xml gui widgets
DEFINES+= QT_FATAL_ASSERT BUILDING_U2TEST_DLL

LIBS += -L../../$$out_dir()
LIBS += -lU2Core$$D -lQSpec$$D
INCLUDEPATH += ../../libs_3rdparty/QSpec/src

if (contains(DEFINES, HI_EXCLUDED)) {
    # GUI testing is not included into public build
    LIBS -= -lQSpec$$D
}
if (!useWebKit()) {
    # GUI testing is available only with WebKit.
    LIBS -= -lQSpec$$D
}

DESTDIR = ../../$$out_dir()

unix {
    target.path = $$UGENE_INSTALL_DIR/
    INSTALLS += target
}

