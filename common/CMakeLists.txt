set(CommonSRC
    CLUtil.h
    CLUtil.cpp
    IComputeTask.h

    Timer.h
    Timer.cpp
    Utils.h
    Utils.cpp
)

add_library(${CommonTarget}
    ${CommonSRC}
)

target_include_directories(${CommonTarget} PUBLIC ${OPENCL_INCLUDE_DIRS})

target_link_libraries(${CommonTarget}
    PRIVATE
    ${OPENCL_LIBRARIES}
)
