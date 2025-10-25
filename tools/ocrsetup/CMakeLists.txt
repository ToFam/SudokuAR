find_package (Qt5 REQUIRED Core Gui Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

SET (OCRSetupSrc
	mainwindow.h
	mainwindow.cpp
	mainwindow.ui
    main.cpp
)

SET (ocrTarget sudokuar-ocrsetup)

add_executable(${ocrTarget}
            ${OCRSetupSrc}
)

target_include_directories(${ocrTarget} PRIVATE ../guicommon)

target_link_libraries(${ocrTarget}
        ${GUICommonTarget}
        ${AlgoTarget}
	Qt5::Widgets
	Qt5::Gui
	Qt5::Core
	${OpenCV_LIBS}
)
