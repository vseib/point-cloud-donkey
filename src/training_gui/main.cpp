// TODO VS temporarily (?) disabling ROS
//#include "ros/ros.h"
#include "training_gui.h"

#include <QApplication>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    // TODO VS temporarily (?) disabling ROS
    //ros::init(argc, argv, "ism_training_gui");

    TrainingGUI gui;
    gui.show();
    app.exec();
    return 0;
}
