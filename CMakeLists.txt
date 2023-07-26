# 基础信息
cmake_minimum_required(VERSION 3.10)
project(rookie)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "DEBUG")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置可执行文件输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 批量处理需要编译的文件
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/image_segment/*.cpp
                    )
foreach (SOURCE ${SOURCES})
    # 获取文件名
    get_filename_component(SOURCE_NAME ${SOURCE} NAME_WE)
    # 添加可执行文件
    add_executable(${SOURCE_NAME} ${SOURCE})
    # 链接OpenCV库
    target_link_libraries(${SOURCE_NAME} ${OpenCV_LIBS})
endforeach (SOURCE)
