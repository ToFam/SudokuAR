find_package (Qt5 REQUIRED Core Gui Widgets)

find_package (pylon 8.0.1 REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

SET (DevelSrc
    AlgoFactory.h
    AlgoFactory.cpp
    main.cpp
    DevelGUIWindow.h
    DevelGUIWindow.cpp
    DevelGUIWindow.ui
	SequenceProvider.h
	SequenceProvider.cpp
	OpenDialog.h
	OpenDialog.cpp
    ContainerList.h
    ContainerList.cpp
    GUIAlgoElement.h
    GUIAlgoElement.cpp
    ContainerDisplay.h
    ContainerDisplay.cpp
    res.qrc
    PylonDriver.h
    PylonDriver.cpp
    
    gui-algo/HoughLinesDisplay.h
    gui-algo/HoughLinesDisplay.cpp
    gui-algo/GridDetectDisplay.h
    gui-algo/GridDetectDisplay.cpp
    gui-algo/TemplateMatchedDisplay.h
    gui-algo/TemplateMatchedDisplay.cpp
)

SET (DevelTarget sudokuar-develgui)

add_executable(${DevelTarget}
               ${DevelSrc}
)

target_include_directories(${DevelTarget} PRIVATE ../guicommon ../../algo)

target_link_libraries(${DevelTarget}
        ${GUICommonTarget}
        ${AlgoTarget}
	Qt5::Widgets
	Qt5::Gui
	Qt5::Core
        opencv_core opencv_video opencv_videoio
        pylon::pylon
)
