# File generated by kdevelop's qmake manager. 
# ------------------------------------------- 
# Subdir relative project main directory: ./src
# Target is an application:  ../bin/smithwaterman

HEADERS += inout.h \
           blosum.h \
           smithwaterman.h \
           jobdirector.h \
           hardwarearchitecturenet.h \
           hardwarearchitecturecuda.h \
           hardwarearchitecturecpu.h \
           swsse2_def.h \
           bioconfig.h \
           hardwarearchitecturecudaprof.h \
           hardwarearchitecturecpusse2.h \
           sw_cpu.h \
 hardwarearchitecturecpuprof.h \
 hardwarearchitecturecudatxt.h
SOURCES += inout.cpp \
           main.cpp \
           smithwaterman.cpp \
           jobdirector.cpp \
           hardwarearchitecturenet.cpp \
           hardwarearchitecturecuda.cpp \
           hardwarearchitecturecpu.cpp \
           bioconfig.cpp \
           hardwarearchitecturecudaprof.cpp \
           hardwarearchitecturecpusse2.cpp \
           sw_cpu.cpp 
mtune = prescott
QT = core 

QMAKE_CXXFLAGS_RELEASE += -O3 \
                          -msse2 
QMAKE_CXXFLAGS_DEBUG += -msse2 
CONFIG += warn_on \
          qt \
          thread \
          exceptions \
          stl \
 debug
TEMPLATE = app 


TARGET = ../bin/swcuda


LIBS += -L../../gpubiolib/ \
  -L../../../elaidecpp/bin \
  -L/usr/local/cuda/lib \
  -L../../cudakernels/ \
  -L/usr/local/cuda/SDK/lib/ \
  -lgpubiolib \
  -lkrlcuda \
  -lGLU \
  -lGL \
  -lcutil \
  -lcudart \
  -lcuda

INCLUDEPATH += ../../ \
  /usr/local/cuda/SDK/common/inc \
  /usr/local/cuda/include


