rm -f *.a
# the main implementation of Smith-Waterman
nvcc -o swhandler4_cu.o -c swhandler4.cu -Xcompiler "-m32" -Xcompiler "" -I. -I$QTINC -I/usr/local/cuda/include -I/usr/local/cuda/SDK/common/inc -DUNIX -O3

#/usr/local/cuda/bin/nvcc -o swhandler6_cu.o -c swhandler6.cu -Xcompiler "-m32" -Xcompiler "" -I. -I$QTINC -I/usr/local/cuda/include -I/usr/local/cuda/SDK/common/inc -DUNIX -O3
#/usr/local/cuda/bin/nvcc -o swhandler7_cu.o -c swhandler7.cu -Xcompiler "-m32" -Xcompiler "" -I. -I$QTINC -I/usr/local/cuda/include -I/usr/local/cuda/SDK/common/inc -DUNIX -O3

# Smith-Waterman kernel implementation with query profile
nvcc -o swhandlerprof_cu.o -c swhandlerprof.cu -Xcompiler "-m32" -Xcompiler "" -I. -I$QTINC -I/usr/local/cuda/include -I/usr/local/cuda/SDK/common/inc -DUNIX -O3

echo ""
echo "Compiling Est2Genome - Solexa ...."
echo ""
# SCAN implementation of GEAGpu
nvcc -o e2g_hndlscan.o -c e2g_hndlscan.cu -arch="compute_11" -code="compute_11" -Xcompiler "-m32" -Xcompiler "" -I. -I$QTINC -I/usr/local/cuda/include -I/usr/local/cuda/SDK/common/inc -DUNIX -O3

# Est2Genome implementation for solexa queries
#/usr/local/cuda/bin/nvcc -o handler_solexa_cu.o -c handler_solexa.cu -Xcompiler "-m32" -Xcompiler "" -I. -I$QTINC -I/usr/local/cuda/include -I/usr/local/cuda/SDK/common/inc -DUNIX -O3

ar r libkrlcuda.a swhandler4_cu.o swhandlerprof_cu.o e2g_hndlscan.o

rm -f *.o
#cd ..
#cd gpubiolib
#rm libgpubiolib.a 
#make

