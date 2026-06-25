cmake_minimum_required(VERSION 3.16)
project(video_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(video_engine src/main.cpp)
target_link_libraries(video_engine PRIVATE ${OpenCV_LIBS})

enable_testing()
add_executable(motion_detector_test tests/test_motion.cpp)
target_link_libraries(motion_detector_test PRIVATE ${OpenCV_LIBS})
target_include_directories(motion_detector_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME motion_detector_test COMMAND motion_detector_test)
