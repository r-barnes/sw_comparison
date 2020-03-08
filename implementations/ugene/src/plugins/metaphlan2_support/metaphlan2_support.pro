include (metaphlan2_support.pri)

# Input
HEADERS += src/Metaphlan2LogParser.h \
           src/Metaphlan2Prompter.h \
           src/Metaphlan2Support.h \
           src/Metaphlan2SupportPlugin.h \
           src/Metaphlan2Task.h \
           src/Metaphlan2Validator.h \
           src/Metaphlan2Worker.h \
           src/Metaphlan2WorkerFactory.h

SOURCES += src/Metaphlan2LogParser.cpp \
           src/Metaphlan2Prompter.cpp \
           src/Metaphlan2Support.cpp \
           src/Metaphlan2SupportPlugin.cpp \
           src/Metaphlan2Task.cpp \
           src/Metaphlan2Validator.cpp \
           src/Metaphlan2Worker.cpp \
           src/Metaphlan2WorkerFactory.cpp