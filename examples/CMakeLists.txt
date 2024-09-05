# 只有安装了opencv才去生成一些工具
include_directories(../src/interface)

add_executable(cas_example_image rb_cas_example_image.cpp rb_cas_example_utils.cpp)
# add_executable(cas_example_video rb_cas_example_video.cpp rb_cas_example_utils.cpp)

target_link_libraries(cas_example_image ${CMAKE_PROJECT_NAME} ${EXTRA_LIBS} -fopenmp -lpthread -lm)
# target_link_libraries(cas_example_video ${CMAKE_PROJECT_NAME} ${EXTRA_LIBS} -fopenmp -lpthread -lm)

#install (TARGETS cas_example_image RUNTIME DESTINATION examples)
# install (TARGETS cas_example_video RUNTIME DESTINATION examples)

FILE(GLOB IMS ../assets/images/*.*)
install(FILES ${IMS} DESTINATION examples)

# 临时测试时使用
install(FILES ${IMS} DESTINATION ${PROJECT_SOURCE_DIR}/temp_example)
#install (TARGETS cas_example_image RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/temp_example)
# install (TARGETS cas_example_video RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/temp_example)

if (WITH_NCNN)
    install(FILES "../assets/models/cas.ncnn.param" DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/models)
    install(FILES "../assets/models/cas.ncnn.bin" DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/models)
elseif (WITH_RKNN)
    install(FILES "../assets/models/cas.rknn" DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/models)
    install(PROGRAMS ${RKNN_RT_LIB} DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/lib)
    install(PROGRAMS ${RGA_LIB} DESTINATION ${PROJECT_SOURCE_DIR}/temp_example/lib)
endif()
