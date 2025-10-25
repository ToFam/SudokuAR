find_package (Qt5 REQUIRED Core Gui Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

SET (GUICommonSrc
	imageview.h
	imageview.cpp
	qimage_util.h
	qimage_util.cpp
)

add_library(${GUICommonTarget}
            ${GUICommonSrc}
)

target_link_libraries(${GUICommonTarget}
	Qt5::Widgets
	Qt5::Gui
	Qt5::Core
	${OpenCV_LIBS}
)