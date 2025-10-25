SET (TestSrc
    Test.h
    Test.cpp
    TestGridDetect.h
    TestGridDetect.cpp
    TestTemplateMatch.h
    TestTemplateMatch.cpp
    main.cpp
)

SET (TestTarget sudokuar)

add_executable(${TestTarget}
            ${TestSrc}
)

target_include_directories(${TestTarget} PRIVATE ../common ../algo)

target_link_libraries(${TestTarget}
    PRIVATE
    ${CommonTarget}
    ${AlgoTarget}
    ${GPUCommonTarget}
    ${OPENCL_LIBRARIES}
    ${OpenCV_LIBS}
)
