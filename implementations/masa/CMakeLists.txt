cmake_minimum_required(VERSION 3.9)

project(cudalign LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_library(masa-core
  masa-core/masa-core-1.3.9.1024/src/common/Job.cpp
  masa-core/masa-core-1.3.9.1024/src/common/CrosspointsFile.cpp
  masa-core/masa-core-1.3.9.1024/src/common/exceptions/IllegalArgumentException.cpp
  masa-core/masa-core-1.3.9.1024/src/common/exceptions/IOException.cpp
  masa-core/masa-core-1.3.9.1024/src/common/biology/Sequence.cpp
  masa-core/masa-core-1.3.9.1024/src/common/biology/SequenceData.cpp
  masa-core/masa-core-1.3.9.1024/src/common/biology/SequenceModifiers.cpp
  masa-core/masa-core-1.3.9.1024/src/common/biology/SequenceInfo.cpp
  masa-core/masa-core-1.3.9.1024/src/common/biology/Alignment.cpp
  masa-core/masa-core-1.3.9.1024/src/common/biology/AlignmentParams.cpp
  masa-core/masa-core-1.3.9.1024/src/common/biology/AlignmentBinaryFile.cpp
  masa-core/masa-core-1.3.9.1024/src/common/sra/SpecialRow.cpp
  masa-core/masa-core-1.3.9.1024/src/common/sra/SpecialRowFile.cpp
  masa-core/masa-core-1.3.9.1024/src/common/sra/SpecialRowRAM.cpp
  masa-core/masa-core-1.3.9.1024/src/common/sra/FirstRow.cpp
  masa-core/masa-core-1.3.9.1024/src/common/sra/SpecialRowsPartition.cpp
  masa-core/masa-core-1.3.9.1024/src/common/sra/SpecialRowsArea.cpp
  masa-core/masa-core-1.3.9.1024/src/common/Properties.cpp
  masa-core/masa-core-1.3.9.1024/src/common/Timer.cpp
  masa-core/masa-core-1.3.9.1024/src/common/RecurrentTimer.cpp
  masa-core/masa-core-1.3.9.1024/src/common/Status.cpp
  masa-core/masa-core-1.3.9.1024/src/common/BestScoreList.cpp
  masa-core/masa-core-1.3.9.1024/src/common/BlocksFile.cpp
  masa-core/masa-core-1.3.9.1024/src/common/SpecialRowReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/InitialCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/FileCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/FileCellsWriter.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/FileStream.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/BufferedStream.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/BufferedCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/BufferedCellsWriter.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/Buffer2.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/BufferLogger.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/ReversedCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/URLCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/DummyCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/SocketCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/URLCellsWriter.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/DummyCellsWriter.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/SocketCellsWriter.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/TeeCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/io/SplitCellsReader.cpp
  masa-core/masa-core-1.3.9.1024/src/common/AlignerPool.cpp
  masa-core/masa-core-1.3.9.1024/src/common/SpecialRowWriter.cpp
  masa-core/masa-core-1.3.9.1024/src/common/AlignerManager.cpp
  masa-core/masa-core-1.3.9.1024/src/common/utils.cpp
  masa-core/masa-core-1.3.9.1024/src/common/configs/ConfigParser.cpp
  masa-core/masa-core-1.3.9.1024/src/common/configs/Configs.cpp

  masa-core/masa-core-1.3.9.1024/src/libmasa/libmasa.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/libmasaTypes.hpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/capabilities.hpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/aligners/AbstractAligner.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/aligners/AbstractAlignerSafe.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/aligners/AbstractBlockAligner.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/aligners/AbstractDiagonalAligner.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/processors/AbstractBlockProcessor.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/processors/CPUBlockProcessor.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/parameters/BlockAlignerParameters.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/parameters/AbstractAlignerParameters.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/pruning/AbstractBlockPruning.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/pruning/BlockPruningDiagonal.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/pruning/BlockPruningGenericN2.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/utils/AlignerUtils.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/IAligner.hpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/IManager.hpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/IAlignerParameter.hpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/Grid.cpp
  masa-core/masa-core-1.3.9.1024/src/libmasa/Partition.cpp

  masa-core/masa-core-1.3.9.1024/src/masanet/MasaNet.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/Peer.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/PeerList.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/MasaNetStatus.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/Command.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdJoin.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdDiscover.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdUndiscover.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdNotifyScore.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdStatusRequest.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdStatusResponse.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdPeerRequest.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdPeerResponse.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdPeerList.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdCreateRing.cpp
  masa-core/masa-core-1.3.9.1024/src/masanet/command/CmdTestRing.cpp

  masa-core/masa-core-1.3.9.1024/src/stage1/sw_stage1.cpp
  masa-core/masa-core-1.3.9.1024/src/stage2/sw_stage2.cpp
  masa-core/masa-core-1.3.9.1024/src/stage3/sw_stage3.cpp
  masa-core/masa-core-1.3.9.1024/src/stage4/sw_stage4.cpp
  masa-core/masa-core-1.3.9.1024/src/stage5/sw_stage5.cpp
  masa-core/masa-core-1.3.9.1024/src/stage6/sw_stage6.cpp
)
target_include_directories(masa-core
  PUBLIC  masa-core/masa-core-1.3.9.1024/src/
  PRIVATE masa-core/masa-core-1.3.9.1024/src/common
)
target_link_libraries(masa-core PUBLIC Threads::Threads ${CUDA_LIBRARIES})
if(DEFINED GPU_SM_ARCH)
  target_compile_options(masa-core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()
target_compile_definitions(masa-core
PUBLIC
  PACKAGE_STRING="MASA-Serial 1.0.1.1024"
  PACKAGE_VERSION="1.3.9.1024"
)



add_executable(cudalign
  masa-cudalign/masa-cudalign-3.9.1.1024/src/main.cpp
  masa-cudalign/masa-cudalign-3.9.1.1024/src/cuda_util.cpp
  masa-cudalign/masa-cudalign-3.9.1.1024/src/CUDAligner.cpp
  masa-cudalign/masa-cudalign-3.9.1.1024/src/CUDAligner.cu
  masa-cudalign/masa-cudalign-3.9.1.1024/src/CUDAlignerParameters.cpp
)
target_compile_options(cudalign PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-malign-double -fno-strict-aliasing>
  $<$<COMPILE_LANGUAGE:CUDA>:-ftz=true -prec-sqrt=false -prec-div=false -Xcompiler=-malign-double,-fno-strict-aliasing>
)
target_compile_definitions(cudalign PRIVATE COMPILED_CUDA_ARCH="Unknown")
target_include_directories(cudalign PRIVATE masa-cudalign/masa-cudalign-3.9.1.1024/src)
target_link_libraries(cudalign PRIVATE Threads::Threads masa-core ${CUDA_LIBRARIES} cuda)
if(DEFINED GPU_SM_ARCH)
  target_compile_options(cudalign PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()



add_executable(masa-openmp
  masa-openmp/masa-openmp-1.0.1.1024/src/main.cpp
  masa-openmp/masa-openmp-1.0.1.1024/src/OpenMPAligner.cpp
  masa-openmp/masa-openmp-1.0.1.1024/src/OpenMPAlignerParameters.cpp
)
target_compile_options(masa-openmp PRIVATE -malign-double -fno-strict-aliasing)
target_link_libraries(masa-openmp PRIVATE Threads::Threads OpenMP::OpenMP_CXX masa-core)
target_include_directories(masa-openmp PRIVATE masa-openmp/masa-openmp-1.0.1.1024/src)



add_executable(masa-serial
  masa-serial/masa-serial-1.0.1.1024/src/main.cpp
  masa-serial/masa-serial-1.0.1.1024/src/SerialBlockAligner.cpp
  masa-serial/masa-serial-1.0.1.1024/src/SerialBlockAlignerParameters.cpp
)
target_compile_options(masa-serial PRIVATE -malign-double -fno-strict-aliasing)
target_link_libraries(masa-serial PRIVATE Threads::Threads OpenMP::OpenMP_CXX masa-core)
