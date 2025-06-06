#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.spatial.transform import Rotation as R

class TrajectoryMetrics:
    def __init__(self):
        rospy.init_node('trajectory_metrics_node')
        
        # Parameters
        self.error_threshold = rospy.get_param("~error_threshold", 0.5)  # meters
        self.gt_list = []
        self.pred_list = []

        # Synchronized subscribers
        gt_sub = Subscriber('/vehicle/odom_ground_truth', Odometry)
        pred_sub = Subscriber('/vehicle/predicted_odom', Odometry)
        
        self.ats = ApproximateTimeSynchronizer(
            [gt_sub, pred_sub],
            queue_size=500,
            slop=0.05  # 50ms synchronization window
        )
        self.ats.registerCallback(self.sync_callback)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("Trajectory metrics node started")
        rospy.spin()

    def sync_callback(self, gt_msg, pred_msg):
        """Store synchronized odometry pairs"""
        self.gt_list.append(gt_msg)
        self.pred_list.append(pred_msg)

    def on_shutdown(self):
        """Calculate and print metrics when node stops"""
        if len(self.gt_list) < 2 or len(self.pred_list) < 2:
            rospy.logwarn("Insufficient data for metrics calculation")
            return

        gt_positions = self.odoms_to_positions(self.gt_list)
        pred_positions = self.odoms_to_positions(self.pred_list)
        gt_poses = self.odoms_to_poses(self.gt_list)
        pred_poses = self.odoms_to_poses(self.pred_list)

        print("\n==== Trajectory Metrics ====")
        print(f"Absolute Trajectory Error (ATE): {self.compute_ate(gt_positions, pred_positions):.4f} m")
        print(f"Average Displacement Error (ADE): {self.compute_ade(gt_positions, pred_positions):.4f} m")
        print(f"Final Displacement Error (FDE): {self.compute_fde(gt_positions, pred_positions):.4f} m")
        print(f"Relative Pose Error (RPE): {self.compute_rpe(gt_poses, pred_poses):.4f} m")
        print(f"Ground Truth Trajectory Accuracy: 100%")
        print(f"Predicted Accuracy (<{self.error_threshold}m): {self.compute_accuracy(gt_positions, pred_positions):.2f}%")
        print("===========================")

    @staticmethod
    def odoms_to_positions(odom_msgs):
        """Convert Odometry messages to numpy array of positions"""
        return np.array([[odom.pose.pose.position.x, 
                        odom.pose.pose.position.y,
                        odom.pose.pose.position.z] 
                      for odom in odom_msgs])

    @staticmethod 
    def odoms_to_poses(odom_msgs):
        """Convert Odometry messages to 4x4 transformation matrices"""
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
    def compute_ate(gt, pred):
        """Absolute Trajectory Error (RMSE)"""
        return np.sqrt(np.mean(np.sum((gt - pred)**2, axis=1)))

    @staticmethod
    def compute_ade(gt, pred):
        """Average Displacement Error"""
        return np.mean(np.linalg.norm(gt - pred, axis=1))

    @staticmethod
    def compute_fde(gt, pred):
        """Final Displacement Error"""
        return np.linalg.norm(gt[-1] - pred[-1])

    @staticmethod
    def compute_rpe(gt_poses, pred_poses):
        """Relative Pose Error (RMSE over relative motions)"""
        errors = []
        for i in range(1, len(gt_poses)):
            gt_rel = np.linalg.inv(gt_poses[i-1]) @ gt_poses[i]
            pred_rel = np.linalg.inv(pred_poses[i-1]) @ pred_poses[i]
            errors.append(np.linalg.norm(gt_rel[:3,3] - pred_rel[:3,3]))
        return np.sqrt(np.mean(np.array(errors)**2))

    def compute_accuracy(self, gt, pred):
        """Percentage of points under error threshold"""
        distances = np.linalg.norm(gt - pred, axis=1)
        return 100 * np.sum(distances < self.error_threshold) / len(distances)

if __name__ == '__main__':
    try:
        TrajectoryMetrics()
    except rospy.ROSInterruptException:
        pass
