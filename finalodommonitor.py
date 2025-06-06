#!/usr/bin/env python3
import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import tf
import tf2_ros
import geometry_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import pyproj

class GPSToENUConverter:
    def __init__(self, lat0, lon0, alt0):
        self.lat0 = lat0
        self.lon0 = lon0
        self.alt0 = alt0

        self.geodetic = pyproj.Proj(proj='latlong', datum='WGS84')
        self.ecef = pyproj.Proj(proj='geocent', datum='WGS84')

        self.geodetic_to_ecef = pyproj.Transformer.from_proj(self.geodetic, self.ecef)
        self.ecef_to_geodetic = pyproj.Transformer.from_proj(self.ecef, self.geodetic)

        self.x0, self.y0, self.z0 = self.geodetic_to_ecef.transform(self.lon0, self.lat0, self.alt0)

    def geodetic_to_enu(self, lat, lon, alt):
        x, y, z = self.geodetic_to_ecef.transform(lon, lat, alt)
        dx = x - self.x0
        dy = y - self.y0
        dz = z - self.z0

        lat0_rad = np.radians(self.lat0)
        lon0_rad = np.radians(self.lon0)

        east = -np.sin(lon0_rad)*dx + np.cos(lon0_rad)*dy
        north = -np.sin(lat0_rad)*np.cos(lon0_rad)*dx - np.sin(lat0_rad)*np.sin(lon0_rad)*dy + np.cos(lat0_rad)*dz
        up = np.cos(lat0_rad)*np.cos(lon0_rad)*dx + np.cos(lat0_rad)*np.sin(lon0_rad)*dy + np.sin(lat0_rad)*dz

        return east, north, up

class OdomMonitorWithICPAndGPS:
    def __init__(self):
        rospy.init_node('odom_monitor_icp_gps_node')

        self.odom_timeout_ms = 100
        self.odom_timeout = rospy.Duration(self.odom_timeout_ms / 1000.0)

        self.last_odom_time = rospy.Time.now()
        self.odom_missing = False
        self.missing_start_time = None
        self.total_missing_duration = rospy.Duration(0)

        self.prev_cloud = None
        self.global_pose = np.eye(4)
        self.seq = 0
        self.last_odom_pose = np.eye(4)

        self.origin_lat = rospy.get_param('~origin_lat', 13.5840)
        self.origin_lon = rospy.get_param('~origin_lon', 79.9620)
        self.origin_alt = rospy.get_param('~origin_alt', 0.0)
        self.gps_converter = GPSToENUConverter(self.origin_lat, self.origin_lon, self.origin_alt)

        self.gps_initial_offset = None
        self.gps_fixed_offset = (29.8, -2.12, 0.0)

        self.last_lidar_time = None

        self.gt_timestamps = []
        self.gt_positions = []
        self.gps_timestamps = []
        self.gps_positions = []

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.predicted_odom_pub = rospy.Publisher("/vehicle/predicted_odom", Odometry, queue_size=10)
        self.gps_odom_pub = rospy.Publisher("/vehicle/gps_odom", Odometry, queue_size=10)

        rospy.Subscriber("/vehicle/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/lidar_points", PointCloud2, self.lidar_callback)
        rospy.Subscriber("/vehicle/gps", NavSatFix, self.gps_callback)
        rospy.Subscriber("/vehicle/odom_ground_truth", Odometry, self.gt_callback)

        rospy.Timer(rospy.Duration(0.05), self.check_timeout)

        rospy.loginfo("Odometry monitor with ICP and GPS started and running.")
        rospy.spin()

    def odom_callback(self, msg):
        self.last_odom_time = rospy.Time.now()
        self.last_odom_pose = self.odom_msg_to_matrix(msg)

        if self.odom_missing:
            missing_interval = self.last_odom_time - self.missing_start_time
            self.total_missing_duration += missing_interval

            rospy.loginfo(f"Odometry resumed. Last missing: {int(missing_interval.to_sec())} s. Total missing: {int(self.total_missing_duration.to_sec())} s")
            self.odom_missing = False
            self.missing_start_time = None
            self.global_pose = np.array(self.last_odom_pose)
            self.prev_cloud = None

        if not rospy.is_shutdown():
            self.predicted_odom_pub.publish(msg)
            self.publish_odom_tf(msg)

    def publish_odom_tf(self, odom_msg):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def check_timeout(self, event):
        if (rospy.Time.now() - self.last_odom_time) > self.odom_timeout:
            if not self.odom_missing:
                self.odom_missing = True
                self.missing_start_time = rospy.Time.now()
                rospy.logwarn("Odometry missing! Switching to ICP prediction mode.")
                self.global_pose = np.array(self.last_odom_pose)
                self.prev_cloud = None

    def lidar_callback(self, msg):
        if not self.odom_missing:
            self.last_lidar_time = msg.header.stamp
            return

        current_time = msg.header.stamp
        if self.last_lidar_time is None:
            self.last_lidar_time = current_time
            return

        dt = (current_time - self.last_lidar_time).to_sec()
        if dt <= 0:
            dt = 0.1

        cloud = self.ros_to_open3d(msg)
        cloud_down = cloud.voxel_down_sample(voxel_size=0.02)

        if self.prev_cloud is None:
            self.prev_cloud = cloud_down
            self.last_lidar_time = current_time
            return

        reg_p2p = o3d.pipelines.registration.registration_icp(
            cloud_down, self.prev_cloud, 0.1, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        transformation = reg_p2p.transformation.copy()
        transformation[0:3, 3] *= dt / 0.1

        self.global_pose = np.dot(self.global_pose, transformation)

        odom_msg = self.pose_to_odom(self.global_pose, self.seq, current_time)
        if not rospy.is_shutdown():
            self.predicted_odom_pub.publish(odom_msg)
        self.seq += 1

        self.prev_cloud = cloud_down
        self.last_lidar_time = current_time

    def gps_callback(self, msg):
        try:
            x, y, z = self.gps_converter.geodetic_to_enu(msg.latitude, msg.longitude, msg.altitude)
        except Exception as e:
            rospy.logerr(f"Error converting GPS to ENU: {e}")
            return

        if self.gps_initial_offset is None:
            self.gps_initial_offset = (x, y, z)
            rospy.loginfo(f"Set GPS initial ENU offset: {self.gps_initial_offset}")

        x -= self.gps_initial_offset[0]
        y -= self.gps_initial_offset[1]
        z -= self.gps_initial_offset[2]

        x += self.gps_fixed_offset[0]
        y += self.gps_fixed_offset[1]
        z += self.gps_fixed_offset[2]

        self.gps_timestamps.append(msg.header.stamp.to_sec())
        self.gps_positions.append([x, y, z])

        if self.gt_timestamps:
            target_time = self.gt_timestamps[-1]
            before = [t for t in self.gps_timestamps if t <= target_time]
            after = [t for t in self.gps_timestamps if t > target_time]

            if before and after:
                t_before = max(before)
                idx_before = self.gps_timestamps.index(t_before)
                pos_before = self.gps_positions[idx_before]

                t_after = min(after)
                idx_after = self.gps_timestamps.index(t_after)
                pos_after = self.gps_positions[idx_after]

                alpha = (target_time - t_before) / (t_after - t_before)
                x = pos_before[0] + alpha * (pos_after[0] - pos_before[0])
                y = pos_before[1] + alpha * (pos_after[1] - pos_before[1])
                z = pos_before[2] + alpha * (pos_after[2] - pos_before[2])

        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation = Quaternion(0, 0, 0, 1)

        if len(msg.position_covariance) == 9:
            cov = msg.position_covariance
            odom.pose.covariance = [
                cov[0], cov[1], cov[2], 0, 0, 0,
                cov[3], cov[4], cov[5], 0, 0, 0,
                cov[6], cov[7], cov[8], 0, 0, 0,
                0, 0, 0, 99999, 0, 0,
                0, 0, 0, 0, 99999, 0,
                0, 0, 0, 0, 0, 99999
            ]
        else:
            odom.pose.covariance = [0.1]*36

        odom.twist.twist.linear.x = 0
        odom.twist.twist.linear.y = 0
        odom.twist.twist.linear.z = 0
        odom.twist.twist.angular.x = 0
        odom.twist.twist.angular.y = 0
        odom.twist.twist.angular.z = 0
        odom.twist.covariance = [-1] + [0]*35

        if not rospy.is_shutdown():
            self.gps_odom_pub.publish(odom)

    def gt_callback(self, msg):
        self.gt_timestamps.append(msg.header.stamp.to_sec())
        self.gt_positions.append([msg.pose.pose.position.x,
                                  msg.pose.pose.position.y,
                                  msg.pose.pose.position.z])
        if len(self.gt_timestamps) > 200:
            self.gt_timestamps.pop(0)
            self.gt_positions.pop(0)

    def ros_to_open3d(self, ros_cloud):
        points_list = []
        for p in pc2.read_points(ros_cloud, skip_nans=True):
            points_list.append([p[0], p[1], p[2]])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_list)
        return cloud

    def pose_to_odom(self, pose_mat, seq, stamp):
        odom = Odometry()
        odom.header.seq = seq
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = pose_mat[0, 3]
        odom.pose.pose.position.y = pose_mat[1, 3]
        odom.pose.pose.position.z = pose_mat[2, 3]

        quat = tf.transformations.quaternion_from_matrix(pose_mat)
        odom.pose.pose.orientation = Quaternion(*quat)

        odom.pose.covariance = [0.0002, 0, 0, 0, 0, 0,
                               0, 0.0002, 0, 0, 0, 0,
                               0, 0, 0.00012, 0, 0, 0,
                               0, 0, 0, 99999, 0, 0,
                               0, 0, 0, 0, 99999, 0,
                               0, 0, 0, 0, 0, 99999]

        odom.twist.twist.linear.x = 0
        odom.twist.twist.linear.y = 0
        odom.twist.twist.linear.z = 0
        odom.twist.twist.angular.x = 0
        odom.twist.twist.angular.y = 0
        odom.twist.twist.angular.z = 0
        odom.twist.covariance = [-1] + [0]*35

        return odom

    def odom_msg_to_matrix(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        trans = np.eye(4)
        trans[0:3, 3] = [pos.x, pos.y, pos.z]
        quat = [ori.x, ori.y, ori.z, ori.w]
        rot = tf.transformations.quaternion_matrix(quat)
        trans[0:3, 0:3] = rot[0:3, 0:3]
        return trans

if __name__ == '__main__':
    try:
        OdomMonitorWithICPAndGPS()
    except rospy.ROSInterruptException:
        pass
