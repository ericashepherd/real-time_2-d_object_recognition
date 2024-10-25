cmake_minimum_required(VERSION 3.0.0)
project(proj3 VERSION 0.1.0)

# links OpenCV & include directories
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE-DIRECTORIES})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

link_libraries(${OpenCV_LIBS})

include_directories(${include})

# generates executables
add_executable(main src/main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)