include (clark_support.pri)

# Input
HEADERS += src/ClarkBuildWorker.h \
           src/ClarkClassifyWorker.h \
           src/ClarkSupport.h \
           src/ClarkSupportPlugin.h \
           src/ClarkTests.h

SOURCES += src/ClarkBuildWorker.cpp \
           src/ClarkClassifyWorker.cpp \
           src/ClarkSupport.cpp \
           src/ClarkSupportPlugin.cpp \
           src/ClarkTests.cpp

TRANSLATIONS += transl/russian.ts
