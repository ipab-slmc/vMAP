#!/usr/bin/env python
import sys
sys.path
sys.path.append('/home/russell/git/vMAP/')
import argparse
import torch
from scipy.spatial.transform import Rotation

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
import message_filters
from cv_bridge import CvBridge

from cfg import Config
import dataset

class DatasetPublisher:

    def __init__(self, args):
        config_file = args.config
        self.cfg = Config(config_file)

        self.bridge = CvBridge()

        self.dataloader = dataset.init_loader(self.cfg)
        self.dataloader_iterator = iter(self.dataloader)
        self.dataset_len = len(self.dataloader)

    def run(self):

        

        rgb_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=10)
        pose_pub = rospy.Publisher('/vicon/camera', PoseStamped, queue_size=10)

        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(1) # 1hz

        counter = 0
        limit = 11
        while not rospy.is_shutdown():

            if counter < limit:

                print(f"publishing image {counter}")

                sample = next(self.dataloader_iterator)

                rgb = sample["image"]
                depth = sample["depth"]
                twc = sample["T"]

                # rbg_cv = torch.transpose(rgb,1,0)

                rgb_msg = self.bridge.cv2_to_imgmsg(rgb.numpy(), encoding="rgb8")
                depth_msg = self.bridge.cv2_to_imgmsg(depth.numpy(), encoding="32FC1")

                rgb_msg.header.stamp = rospy.Time.now()
                depth_msg.header.stamp = rgb_msg.header.stamp

                pose_msg = PoseStamped()
                pose_msg.header.stamp = rgb_msg.header.stamp
                pose_msg.header.frame_id = "world"
                # pose_msg.child_frame_id = "camera"

                pose_rot = Rotation.from_matrix(twc[0:3,0:3])
                pose_quat = pose_rot.as_quat()

                pose_msg.pose.position.x = twc[0,3]
                pose_msg.pose.position.y = twc[1,3]
                pose_msg.pose.position.z = twc[2,3]

                pose_msg.pose.orientation.x = pose_quat[0]
                pose_msg.pose.orientation.y = pose_quat[1]
                pose_msg.pose.orientation.z = pose_quat[2]
                pose_msg.pose.orientation.w = pose_quat[3]

                rgb_pub.publish(rgb_msg)
                depth_pub.publish(depth_msg)
                pose_pub.publish(pose_msg)

                print(pose_msg.pose)

            else:
                break

            counter += 1
            rate.sleep()

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


    dataset_publisher = DatasetPublisher(args)

    try:
        dataset_publisher.run()
    except rospy.ROSInterruptException:
        pass
