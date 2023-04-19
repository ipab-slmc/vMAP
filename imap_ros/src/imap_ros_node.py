#!/usr/bin/env python

import time
import loss
from vmap import *
import utils
import open3d
import matplotlib.pyplot as plt
import dataset
import vis
from functorch import vmap
import argparse
from cfg import Config
import shutil

# ROS imports
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2

import message_filters


class ROSImplicitMapper:
  
  def __init__(self):
    # Publishers
    self.input_pointcloud_pub = rospy.Publisher('input_pointcloud', PointCloud2, queue_size=10)
    self.reconstruction_pub = rospy.Publisher('reconstruction', PointCloud2, queue_size=10)
    # Subscribers
    self.rgb_sub =  message_filters.Subscriber("/camera/image_raw", Image)
    self.depth_sub =  message_filters.Subscriber("/camera/depth/image_raw", Image)

    ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
    ts.registerCallback(self.cameraCallback)
    rospy.spin()

  def cameraCallback(self, rgb_image, depth_image):
    print("Got RGBD image")

    # todo: add to network



if __name__ == '__main__':

    rospy.init_node('imap_ros_node')

    mapper = ROSImplicitMapper()

    rospy.spin()