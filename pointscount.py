#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

class LidarStats:
    def __init__(self):
        self.total_frames = 0
        self.total_points = 0
        rospy.init_node('lidar_points_counter', anonymous=True)
        rospy.Subscriber("/lidar_points", PointCloud2, self.callback_lidar)
        rospy.on_shutdown(self.print_summary)
        rospy.spin()

    def callback_lidar(self, data):
        point_count = sum(1 for _ in pc2.read_points(data, skip_nans=True))
        self.total_frames += 1
        self.total_points += point_count

    def print_summary(self):
        if self.total_frames > 0:
            avg_points = self.total_points / self.total_frames
            print(f"Total frames: {self.total_frames}")
            print(f"Average points per frame: {avg_points:.2f}")
        else:
            print("No frames received.")

if __name__ == '__main__':
    LidarStats()
