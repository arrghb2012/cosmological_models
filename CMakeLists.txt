cmake_minimum_required(VERSION 2.6)
project(mcmc_init_cond_2)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -funroll-loops -march=native -std=c++1y -Wall -Wextra -Wconversion -Wshadow")
find_package(GSL REQUIRED)
find_package(Armadillo)
include_directories(${PROJECT_SOURCE_DIR}/inc ${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})
set(LIBS ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES} ${ARMADILLO_LIBRARIES})
link_libraries(${LIBS})

find_package(Boost 1.54) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 

  endif()

FIND_PACKAGE( OpenMP REQUIRED)

FIND_PACKAGE (HDF5) # Find non-cmake built HDF5
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIR})
set(_hdf5_libs hdf5 hdf5_cpp)

set(main_sources
  src/CosmologicalModel.cpp  
  src/LambdaCDM.cpp
  src/SigmaCDM.cpp
  src/utils.cpp
)

add_executable(main1 src/main.cpp  ${main_sources})
target_link_libraries(main ${_hdf5_libs} ${Boost_LIBRARIES})

include_directories($ENV{GMOCK_HOME}/include $ENV{GMOCK_HOME}/gtest/include)
link_directories($ENV{GMOCK_HOME}/mybuild $ENV{GMOCK_HOME}/gtest/mybuild)

set(sources 
  src/CosmologicalModel.cpp  
  src/LambdaCDM.cpp
  src/SigmaCDM.cpp
  src/utils.cpp
  test/runUnitTests.cpp
  test/testLambdaCDM.cpp)

add_executable(runUnitTests ${sources})
target_link_libraries(runUnitTests gmock gtest pthread ${_hdf5_libs})
