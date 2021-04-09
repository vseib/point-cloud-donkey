// TODO VS temporarily (?) disabling ROS
//#include "ros/ros.h"
#include "model_gui.h"

#include <QApplication>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    // TODO VS temporarily (?) disabling ROS
    //ros::init(argc, argv, "ism_model_gui");

    ModelGUI gui;
    gui.show();
    app.exec();
    return 0;
}
