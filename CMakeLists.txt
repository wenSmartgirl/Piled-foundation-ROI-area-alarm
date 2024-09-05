cmake_minimum_required (VERSION 3.10)

# 获取git的tag版本号
execute_process(COMMAND git describe --tags 
    TIMEOUT 5
    OUTPUT_VARIABLE GIT_TAG_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 获得 md5 值
execute_process(COMMAND git describe --always 
    TIMEOUT 5
    OUTPUT_VARIABLE GIT_COMMIT_MD5
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "building from git tag ${GIT_TAG_VERSION}")
add_definitions(-DBUILD_TAG_VERSION=\"${GIT_TAG_VERSION}\")

# 设置项目名称和版本号
project(CAS)

# 获取当前时间
string(TIMESTAMP BUILDDATETIME "%Y-%m-%d %H:%M:%S")

# 签名信息, 设置签名信息串
set(WEARE "@2023 Hefei Source Intelligence Technology Co,.Ltd."
    CACHE STRING "Who we are and this string will write into signature")
set(CAS_SIGNATURE_STR 
    "${WEARE}, || Project: ${CMAKE_PROJECT_NAME} || Version: ${GIT_TAG_VERSION} || Build DateTime: ${BUILDDATETIME} || Commit-MD5:${GIT_COMMIT_MD5}" CACHE STRING "The signature info string which will add to the target files")


# 设置安装目录
set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install")
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 要求编译器支持C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(NOT WIN32)
#     set(CMAKE_C_FLAGS_RELEASE    "${CMAKE_C_FLAGS_RELEASE} -s")
#     set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -s")
    set(CMAKE_C_FLAGS_RELEASE    "${CMAKE_C_FLAGS_RELEASE} -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
endif()

# 送入网络图像大小
set(TGT_IM_SZ  224)

# 目标阈值
set(PROB_THR  0.25)

# NMS 阈值
set(NMS_THR   0.25)

# TODO: 类别数, rknn 需要, 根据情况进行修改 
set(OBJ_CLASS_NUM 4)
# 最多目标数
set(OBJ_NUMB_MAX_SIZE 64)

######################### 设置内部参数部分 ################################
# TODO: 模型前缀 model prefix, 每次训练导出模型时请记得修改  ***************
set(MODEL_PREFIX   cas_20220928)

# 参数设置
set(BUFFER_LEN 5)

# 内部收缩像素数
set(INNER_BORDER  5)

# 外部扩展像素数
set(OUTTER_BORDER 5)

# 设置目标与ROI重叠阈值(0~1)
set(OVERLAP_THR  0.1)

# 设置低灵敏度的倍数(灵敏度越低，重叠阈值越大 1~2)
set(LOW_SENS_RATIO  1.3)

# 设置高灵敏度的倍数(灵敏度越高，重叠阈值越小, 0~1)
set(HIGH_SENES_RATIO 0.8)
##########################################################################

# 要求编译器支持C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(NOT WIN32)
    set(CMAKE_C_FLAGS_RELEASE    "${CMAKE_C_FLAGS_RELEASE} -s")
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

# 开启相关选项
option (WITH_OPENMP     "Use OpenMP"                                                             ON )
option (WITH_NCNN       "Use ncnn inference "                                                    OFF )
option (WITH_RKNN       "Use rknn npu inference"                                                 ON )
option (LOAD_MEM_MODEL  "Build lib, with load network from memory, else load from file"          OFF )
option (BUILD_DEBUG     "Build Debug, will print or save some temp message"                      ON )
option (BUILD_EXAMPLES  "Build examples"                                                         ON )

# 使用openmp
if (WITH_OPENMP)
    add_definitions(-fopenmp)
endif(WITH_OPENMP)

# 编译类型
if (BUILD_DEBUG)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build types: Release Debug" FORCE)
    option (SHOW_DEBUG_IMG  "Show image inner the api"       ON )   # 显示调试图像
    option (SHOW_DEBUG_TIME "Show process time in the api"   ON )   # 显示内部时间
    option (SHOW_DEBUG_INFO "Show debug info in the api"     ON )   # 显示内部打印
else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build types: Release Debug" FORCE)
    option (SHOW_DEBUG_IMG  "Show image inner the api"     OFF )   # 不显示调试图像
    option (SHOW_DEBUG_TIME "Show process time in the api" OFF )   # 不显示内部时间
    option (SHOW_DEBUG_INFO "Show debug info in the api"   OFF )   # 不显示内部打印
endif()


# OpenCV路径设置
set(OpenCV_DIR "/home/ztl/depends/opencv-4.5.5/build")

 #set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
#set(OpenCV_LIB_DIRS "${OpenCV_DIR}/lib")
set(OpenCV_LIB_DIRS "/usr/local/lib")
set(OpenCV_3RD_DIRS "${OpenCV_DIR}/3rdparty/lib")
include_directories("${OpenCV_INCLUDE_DIRS}")
link_directories("${OpenCV_LIB_DIRS}")
link_directories("${OpenCV_3RD_DIRS}")
list(APPEND EXTRA_LIBS opencv_imgproc opencv_core opencv_highgui opencv_videoio opencv_imgcodecs -fopenmp)

# 设置使用 NCNN
if (WITH_NCNN)
    option (WITH_RKNN       "Use rknn npu inference"   OFF )
    # ncnn路径设置
    set(NCNN_DIR "/home/ztl/app/down/ncnn/build/install") 
    set(NCNN_INCLUDE_DIR "${NCNN_DIR}/include/ncnn")
    set(NCNN_LIB_DIR "${NCNN_DIR}/lib")
    include_directories("${NCNN_INCLUDE_DIR}")
    link_directories("${NCNN_LIB_DIR}")
    list(APPEND EXTRA_LIBS ncnn)
    
    
    if (LOAD_MEM_MODEL)
        message(STATUS "USE NCNN, aand Load memory model!")
    else()
        # 当不用内存方式时, 安装模型文件
        set(NCNN_PARAM_PATH "models/cas.ncnn.param")
        set(NCNN_BIN_PATH   "models/cas.ncnn.bin")
        install(FILES
            "assets/models/cas.ncnn.bin"
            "assets/models/cas.ncnn.param"
            DESTINATION assets/models)
    endif()
elseif(WITH_RKNN)
    option (WITH_NCNN       "Use ncnn inference "      OFF )
    include_directories(${CMAKE_SOURCE_DIR}/../3rdparty)

    # rknn api runtime
    set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../../3rdparty/runtime)
    set(RKNN_RT_LIB ${RKNN_API_PATH}/lib/librknnrt.so)
    include_directories(${RKNN_API_PATH}/include)
    list(APPEND EXTRA_LIBS ${RKNN_RT_LIB})

    # rga
    set(RGA_PATH ${CMAKE_SOURCE_DIR}/../../../3rdparty/rga)
    set(RGA_LIB ${RGA_PATH}/lib/librga.so)
    include_directories(${RGA_PATH}/include)
    list(APPEND EXTRA_LIBS ${RGA_LIB})


    set(MODEL_NAME "models/cas.rknn")
    install(FILES
        "assets/models/cas.rknn"
        DESTINATION models
    )

    install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
    install(PROGRAMS ${RGA_LIB}  DESTINATION lib)

endif()

# windows
if(MSVC)
    add_definitions(/utf-8)
    add_definitions(/MT)
    # add_definitions(/MDd)
    add_definitions(/MD)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DSI_API_EXPORTS)
endif()

# 配置文件到实例目录
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/rb_cas_gconfig.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/rb_cas_gconfig.h"
)

# 配置文件到 module 内部
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/rb_cas_gconfig.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/module/include/rb_cas_gconfig.h"
)

# 生成签名信息串的cpp源文件
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/rb_cas_signature.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rb_cas_signature.cpp"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/rb_cas_signature.cpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/module/source/rb_cas_signature.cpp"
)

# 包含头文件
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/interface")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/module/include")

# 添加源代码目录进行编译
add_subdirectory(src)

# 添加用例
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)


# 调试信息
message(STATUS "---------------- message start -------------------------------")
message(STATUS "Tags: ${GIT_TAG_VERSION}, Commits md5: ${GIT_COMMIT_MD5}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "Building at ${BUILDDATETIME}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "BUILD_SHARED_LIBRARY = ${BUILD_SHARED_LIBRARY}")
message(STATUS "OpenCV Path = ${OpenCV_DIR}")
message(STATUS "NCNN Path = ${NCNN_DIR}")
message(STATUS "RGA Path = ${RGA_PATH}")
message(STATUS "---------------- message end -------------------------------")

# 打包库, 可以用make package
include (InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE
    "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION  ${GIT_TAG_VERSION})
set(CPACK_GENERATOR "TGZ")
include (CPack)
