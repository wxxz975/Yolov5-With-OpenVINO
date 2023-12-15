cmake_minimum_required(VERSION 3.20)
project(yolov5_openvino)

find_package(OpenCV)
find_package(OpenVINO COMPONENTS Runtime)
find_package(gflags)


file (GLOB_RECURSE MAIN_SRC *.cpp)
file (GLOB_RECURSE MAIN_HEADERS *.h)


include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    )

add_executable(
    ${PROJECT_NAME}
    ${MAIN_SRC} 
)

target_link_libraries(
    ${PROJECT_NAME}    
    ${OpenCV_LIBRARIES}
    openvino::runtime
)




