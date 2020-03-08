# include (U2Gui.pri)

MODULE_ID=U2Gui
include( ../../ugene_lib_common.pri )

QT += network xml svg sql widgets printsupport
DEFINES+= QT_FATAL_ASSERT BUILDING_U2GUI_DLL
INCLUDEPATH += ../U2Private/src

useWebKit() {
    QT += webkitwidgets
} else {
    QT += webenginewidgets websockets webchannel
}
LIBS += -L../../$$out_dir()
LIBS += -lU2Core$$D -lU2Formats$$D -lU2Private$$D

DESTDIR = ../../$$out_dir()

        unix {
    target.path = $$UGENE_INSTALL_DIR/
    INSTALLS += target
}
