cmake_minimum_required(VERSION 3.0.2)
project(imap_ros)

set(CATKIN_PACKAGE_DEPENDENCIES
    geometry_msgs
    sensor_msgs
    rospy
)
## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS 
  ${CATKIN_PACKAGE_DEPENDENCIES}
)

catkin_package(
  CATKIN_DEPENDS
    ${CATKIN_PACKAGE_DEPENDENCIES}
)

install(PROGRAMS src/imap_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)