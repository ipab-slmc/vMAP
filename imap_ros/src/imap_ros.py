#!/usr/bin/env python

# Python
import argparse
from scipy.spatial.transform import Rotation
import numpy as np
import open3d

# PyTorch
import torch
import torch.multiprocessing as mp

# ROS imports
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
import message_filters
from cv_bridge import CvBridge


# vMap
from IMap import IMapWrapper


# Bit operations
BIT_MOVE_16 = int(2**16)
BIT_MOVE_8 = int(2**8)

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]


# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
# https://github.com/felixchenfy/open3d_ros_pointcloud_conversion
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points = np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors:  # XYZ only
        fields = FIELDS_XYZ
        cloud_data = points
    else:  # XYZ + RGB
        fields = FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255).astype(int)  # nx3 matrix
        colors = colors[:, 0] * BIT_MOVE_16 + colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
        colors = colors.reshape(colors.shape[0], 1)
        cloud_data = np.concatenate((points, colors), axis=1, dtype=object)
        # cloud_data=np.c_[points, colors, dtype='f4, f4, f4, i4']

    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)


class ROSImplicitMapper:

    def __init__(self, args):

        self.camera_rate = 1.0  # Hz
        self.add_image_rate = 1.0  # Hz
        self.image_counter = 0
        self.args = args

        self.mapper = IMapWrapper(self.args)

        self.bridge = CvBridge()

        # Publishers
        self.input_pointcloud_pub = rospy.Publisher('input_pointcloud', PointCloud2, queue_size=10)
        self.reconstruction_pub = rospy.Publisher('reconstruction', PointCloud2, queue_size=10)
        # Subscribers
        self.rgb_sub = message_filters.Subscriber("/camera/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)

        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        ts.registerCallback(self.cameraCallback)

        rospy.Subscriber("/vicon/camera", PoseStamped, self.viconPoseCallback)
        self.latest_pose = PoseStamped()  # might need to improve this with mutex or queue

    def run(self):

        rate = rospy.Rate(1)  # 10hz

        while not rospy.is_shutdown():

            self.mapper.meshing()

            open3d_mesh = self.mapper.getMesh()

            if (open3d_mesh is not None):
                print("Mesh available ")
                pc = open3d.geometry.PointCloud()
                pc.points = open3d_mesh.vertices
                pc.colors = open3d_mesh.vertex_colors
                ros_msg = convertCloudFromOpen3dToRos(pc, frame_id="world")
                ros_msg.header.stamp = rospy.Time.now()
                print("Publishing Reconstruction Data")
                self.reconstruction_pub.publish(ros_msg)
            else:
                # print("Mesh not available yet")
                pass

            rate.sleep()

    def cameraCallback(self, rgb_image, depth_image):

        print(f" Got RGBD image {self.image_counter}")

        if (self.image_counter % (self.camera_rate / self.add_image_rate) == 0):
            print(f"Processing RGBD image {self.image_counter}")

            # convert images and pose to tensors
            rgb_open_cv = self.bridge.imgmsg_to_cv2(rgb_image, desired_encoding=rgb_image.encoding)
            depth_open_cv = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding=depth_image.encoding)
            rgb_image_tensor = torch.tensor(rgb_open_cv)
            depth_image_tensor = torch.tensor(depth_open_cv)

            rotation_mat = Rotation.from_quat(
                [self.latest_pose.pose.orientation.x, self.latest_pose.pose.orientation.y, self.latest_pose.
                 pose.orientation.z, self.latest_pose.pose.orientation.w])

            pose_tensor = torch.eye(4, 4)
            pose_tensor[0:3, 0:3] = torch.tensor(rotation_mat.as_matrix())
            pose_tensor[0, 3] = self.latest_pose.pose.position.x
            pose_tensor[1, 3] = self.latest_pose.pose.position.y
            pose_tensor[2, 3] = self.latest_pose.pose.position.z

            print(f"Pose: \n {pose_tensor}")

            # Add measurements to network
            self.mapper.add_images_and_pose(rgb_image_tensor, depth_image_tensor, pose_tensor)
        else:
            pass

        self.image_counter += 1

    def viconPoseCallback(self, msg):
        print("Got vicon pose")
        self.latest_pose = msg


if __name__ == '__main__':

    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--logdir', default='./logs/debug',
                        type=str)
    parser.add_argument('--config',
                        default='/home/russell/git/vMAP/configs/Replica/config_replica_room0_iMAP.json',
                        type=str)
    parser.add_argument('--save_ckpt',
                        default=False,
                        type=bool)
    args = parser.parse_args()

    rospy.init_node('imap_ros')

    # mp.set_start_method('spawn')  # Required for m/ultiprocessing on some platforms

    rosMapper = ROSImplicitMapper(args)

    try:
        rosMapper.run()
    except KeyboardInterrupt:
        print('interrupted!')
