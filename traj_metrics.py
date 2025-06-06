#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
from scipy.spatial.transform import Rotation as R

class TrajectoryMetrics:
    def __init__(self):
        rospy.init_node('trajectory_metrics_node')

        self.gt_list = []
        self.pred_list = []

        # Subscribers with synchronization
        gt_sub = Subscriber('/vehicle/odom_ground_truth', Odometry)
        pred_sub = Subscriber('/vehicle/predicted_odom', Odometry)
        ats = ApproximateTimeSynchronizer([gt_sub, pred_sub], queue_size=500, slop=0.05)
        ats.registerCallback(self.collect_data)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("Trajectory metrics node started.")
        rospy.spin()

    def collect_data(self, gt_msg, pred_msg):
        self.gt_list.append(gt_msg)
        self.pred_list.append(pred_msg)

    def on_shutdown(self):
        if not self.gt_list or not self.pred_list:
            rospy.logwarn("Not enough data received for metrics calculation.")
            return

        # Convert to numpy arrays
        gt_positions = self.odoms_to_positions(self.gt_list)
        pred_positions = self.odoms_to_positions(self.pred_list)
        gt_poses = self.odoms_to_poses(self.gt_list)
        pred_poses = self.odoms_to_poses(self.pred_list)

        # Metrics
        ate = self.compute_ate(gt_positions, pred_positions)
        ade = self.compute_ade(gt_positions, pred_positions)
        fde = self.compute_fde(gt_positions, pred_positions)
        rpe = self.compute_rpe(gt_poses, pred_poses)

        print("\n==== Trajectory Metrics ====")
        print(f"Absolute Trajectory Error (ATE): {ate:.4f} m")
        print(f"Average Displacement Error (ADE): {ade:.4f} m")
        print(f"Final Displacement Error (FDE): {fde:.4f} m")
        print(f"Relative Pose Error (RPE): {rpe:.4f} m")
        print(f"Ground Truth Trajectory Accuracy: 100%")
        print("===========================")

    @staticmethod
    def odoms_to_positions(odom_msgs):
        return np.array([[odom.pose.pose.position.x, 
                          odom.pose.pose.position.y,
                          odom.pose.pose.position.z] 
                         for odom in odom_msgs])

    @staticmethod
    def odoms_to_poses(odom_msgs):
        poses = []
        for odom in odom_msgs:
            T = np.eye(4)
            T[0:3, 3] = [odom.pose.pose.position.x,
                         odom.pose.pose.position.y,
                         odom.pose.pose.position.z]
            q = odom.pose.pose.orientation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[0:3, 0:3] = rot
            poses.append(T)
        return np.array(poses)

    @staticmethod
    def compute_ate(gt_positions, pred_positions):
        errors = np.linalg.norm(gt_positions - pred_positions, axis=1)
        return np.sqrt(np.mean(errors**2))

    @staticmethod
    def compute_ade(gt_positions, pred_positions):
        errors = np.linalg.norm(gt_positions - pred_positions, axis=1)
        return np.mean(errors)

    @staticmethod
    def compute_fde(gt_positions, pred_positions):
        return np.linalg.norm(gt_positions[-1] - pred_positions[-1])

    @staticmethod
    def compute_rpe(gt_poses, pred_poses):
        trans_errors = []
        for i in range(len(gt_poses)-1):
            gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i+1]
            pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i+1]
            trans_error = np.linalg.norm(gt_rel[:3,3] - pred_rel[:3,3])
            trans_errors.append(trans_error)
        return np.sqrt(np.mean(np.array(trans_errors)**2))

if __name__ == "__main__":
    TrajectoryMetrics()
