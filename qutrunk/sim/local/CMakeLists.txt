CMAKE_MINIMUM_REQUIRED(VERSION 3.7)
PROJECT(simcpp VERSION 1.0)

IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(LIBRARY_OUTPUT_PATH ../)
ELSE ()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(CMAKE_BUILD_TYPE "Release")
  set(LIBRARY_OUTPUT_PATH ../Release)
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Windows")

set(CMAKE_CXX_STANDARD 17)

set(SIM_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/cppsim.cpp
)
 
include_directories(
  ${PROJECT_SOURCE_DIR}/pybind11/include
  ${PROJECT_SOURCE_DIR}/QuEST/include
)

add_subdirectory(pybind11)
add_subdirectory(QuEST)

pybind11_add_module(simulator ${SIM_SOURCE_FILES})

target_link_libraries(simulator PRIVATE
    QuEST
)