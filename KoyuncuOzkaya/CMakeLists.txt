cmake_minimum_required(VERSION 3.10)
project(octomap_pybind_example)


find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(extern/pybind11)  # will build pybind11 from that source
add_library(octomap_pybind SHARED octomap_pybind.cpp)
target_link_libraries(octomap_pybind PRIVATE pybind11::module)


target_link_libraries(octomap_pybind 
    PRIVATE 
        pybind11::module 
        ${OpenCV_LIBS} 
        ${OCTOMAP_LIBRARIES}
)
set_target_properties(octomap_pybind PROPERTIES 
    PREFIX ""
    SUFFIX ".so"
)
