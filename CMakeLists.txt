cmake_minimum_required(VERSION 3.0.0)
project(cvt VERSION 0.1.0)
 
include(CTest)
enable_testing()
 
find_package(OpenCV REQUIRED)
add_executable(cvt main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cvt ${OpenCV_LIBS})
 
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
 
 
