SOURCES += handler_solexa.cpp \
 biosequence.cpp \
 e2gengine.cpp \
 alignmentpair.cpp \
 shortsequencescache.cpp \
 ConfigFile.cpp

TEMPLATE = lib

CONFIG += staticlib

QT -= gui


HEADERS += handler_solexa.h \
 biosequence.h \
 e2gengine.h \
 alignmentpair.h \
 shortsequencescache.h \
 ConfigFile.h




INCLUDEPATH += ../cudakernels/ \
  /usr/local/cuda/include \
  ../../../NVIDIA_CUDA_SDK/common/inc

LIBS += -L../cudakernels/ \
  -lswcuda


QMAKE_CXXFLAGS_RELEASE += -O3

