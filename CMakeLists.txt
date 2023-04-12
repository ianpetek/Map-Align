cmake_minimum_required(VERSION 3.23)
project(map_align)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV 4.1.2 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(map_align
        main.cpp)
target_include_directories(map_align PRIVATE ${OpenCV_INCLUDE_DIRS} yaml-cpp)
target_link_libraries(map_align Eigen3::Eigen ${OpenCV_LIBS} yaml-cpp)