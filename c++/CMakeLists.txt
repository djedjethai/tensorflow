cmake_minimum_required(VERSION 3.10)
project(tensorflow)

set(TENSORFLOW_PATH "/home/jerome/Documents/projects/smartDentalPictures/tensorflow/c++")
set(TENSORFLOW_HEADERS "/home/jerome/Documents/projects/smartDentalPictures/tensorflow/c++/include")
set(OPENCV_PATH "/home/jerome/clibs/opencvBUILT/lib" CACHE PATH "Path to openCV")
set(OPENCV_HEADERS "/home/jerome/clibs/opencvBUILT/include/opencv4" CACHE PATH "Path to openCV-Headers")


if(NOT TENSORFLOW_PATH)
	message(FATAL_ERROR "Missing tensorflow install path, please specify TENSORFLOW_PATH")
else()
	message("tensorflow path is ${TENSORFLOW_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Source files
set(SOURCES 
	src/main.cpp
)

add_executable(tensorflow ${SOURCES})

# include headers
target_include_directories(tensorflow PRIVATE ${TENSORFLOW_HEADERS} ${OPENCV_HEADERS})

# link libraries
target_link_libraries(tensorflow PRIVATE 
	${OPENCV_PATH}/libopencv_core.so
	${OPENCV_PATH}/libopencv_imgcodecs.so
	${OPENCV_PATH}/libopencv_imgproc.so
	${TENSORFLOW_PATH}/lib/libtensorflow.so.2
	${TENSORFLOW_PATH}/lib/libtensorflow_framework.so.2
)



# # ===============================================
# # Include TensorFlow headers
# include_directories(${TENSORFLOW_PATH}/include)
# 
# # Link TensorFlow libraries
# link_directories(${TENSORFLOW_PATH}/lib)
# 
# # Add executable
# 
# # Link TensorFlow library
# target_link_libraries(TensorFlowInference tensorflow)
# # target_link_libraries(TensorFlowInference ${TENSORFLOW_PATH}/lib/libtensorflow.so)
# 
