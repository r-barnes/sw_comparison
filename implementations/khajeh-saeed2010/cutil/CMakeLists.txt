# find_package (GLUT REQUIRED)
# find_package (GLEW REQUIRED)
find_package(CUDA)

set (BUILD_SHARED_LIBS OFF)

add_library(cutil
  src/bank_checker.cpp
  src/cmd_arg_reader.cpp
  src/cutil.cpp
  src/stopwatch.cpp
  src/stopwatch_linux.cpp
  src/multithreading.cpp
)

target_include_directories(cutil PUBLIC inc)
# target_link_libraries(cutil PUBLIC cudart)

# ADD_LIBRARY (paramgl
#   src/paramgl.cpp
#   src/param.cpp
# )

# ADD_LIBRARY (rendercheckgl
  # src/rendercheck_gl.cpp
# )

# if (WIN32)
#   SET_TARGET_PROPERTIES(cutil PROPERTIES DEFINE_SYMBOL BUILD_DLL)
# endif (WIN32)

# TARGET_LINK_LIBRARIES(paramgl
#   ${GLUT_LIBRARIES}
#   ${GLEW_LIBRARY}
# )

# TARGET_LINK_LIBRARIES(rendercheckgl
#   cutil
#   ${GLUT_LIBRARIES}
#   ${GLEW_LIBRARY}
# )