include(../ugene_globals.pri)

QT += xml network widgets
TEMPLATE = app
CONFIG += qt thread debug_and_release
macx : CONFIG -= app_bundle
DEFINES+= QT_DLL QT_FATAL_ASSERT
INCLUDEPATH += src _tmp

DESTDIR = ../$$out_dir()

!debug_and_release|build_pass {

    CONFIG(debug, debug|release) {
        TARGET = ugenem
        DEFINES+=_DEBUG
        CONFIG +=console
        MOC_DIR=_tmp/moc/debug
        OBJECTS_DIR=_tmp/obj/debug
    }

    CONFIG(release, debug|release) {
        TARGET = ugenem
        DEFINES+=NDEBUG
        MOC_DIR=_tmp/moc/release
        OBJECTS_DIR=_tmp/obj/release
    }
}

UI_DIR=_tmp/ui
RCC_DIR=_tmp/rcc

win32 {

    LIBS += -luser32     # to import CharToOemA with nmake build

    QMAKE_CXXFLAGS_WARN_ON = -W3
    QMAKE_CFLAGS_WARN_ON = -W3

    RC_FILE = ugenem.rc
}

unix {
    target.path = $$UGENE_INSTALL_DIR/
    INSTALLS += target
}

unix_not_mac() : LIBS += -lX11

HEADERS += src/DetectWin10.h \
           src/SendReportDialog.h \
           src/Utils.h 

FORMS += src/ui/SendReportDialog.ui

SOURCES += src/DetectWin10.cpp \
           src/main.cpp \
           src/SendReportDialog.cpp \
           src/Utils.cpp \
           src/getMemorySize.c

RESOURCES += ugenem.qrc
