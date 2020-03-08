# include (U2Formats.pri)

MODULE_ID=U2Formats
include( ../../ugene_lib_common.pri )

LIBS += $$add_z_lib()

DEFINES += QT_FATAL_ASSERT BUILDING_U2FORMATS_DLL

LIBS += -L../../$$out_dir()
LIBS += -lU2Core$$D -lU2Algorithm$$D
LIBS += -lsamtools$$D
LIBS += $$add_sqlite_lib()

win32-msvc2013 {
    DEFINES += NOMINMAX _XKEYCHECK_H
}

QT += sql widgets

# Force re-linking when lib changes
unix:POST_TARGETDEPS += ../../$$out_dir()/libsamtools$${D}.a
# Same options which samtools is built with
DEFINES+="_FILE_OFFSET_BITS=64" _LARGEFILE64_SOURCE _USE_KNETFILE
INCLUDEPATH += ../../libs_3rdparty/samtools/src ../../libs_3rdparty/samtools/src/samtools
win32:INCLUDEPATH += ../../libs_3rdparty/samtools/src/samtools/win32
win32:LIBS += -lws2_32
win32:DEFINES += _USE_MATH_DEFINES "__func__=__FUNCTION__" "R_OK=4" "atoll=_atoi64" "alloca=_alloca"

win32 {
    # not visual studio 2015
    !win32-msvc2015 {
        DEFINES += "inline=__inline"
    }
}

INCLUDEPATH += ../../libs_3rdparty/sqlite3/src
DESTDIR = ../../$$out_dir()

unix {
    target.path = $$UGENE_INSTALL_DIR/
    INSTALLS += target
}
