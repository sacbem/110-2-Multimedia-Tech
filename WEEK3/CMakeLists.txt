cmake_minimum_required(VERSION 3.0.0)
project(WEEK3 VERSION 0.1.0)
 
include(CTest)
enable_testing()
 
find_package(OpenCV REQUIRED)
add_executable(WEEK3 main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(WEEK3 ${OpenCV_LIBS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
 