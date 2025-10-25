#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    DevelGUIWindow w;

    if (argc == 2)
        w.load(argv[1]);

    w.show();

    return a.exec();
}
