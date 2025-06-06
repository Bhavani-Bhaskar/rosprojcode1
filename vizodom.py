#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

class OdomVisualizer:
    def __init__(self):
        rospy.init_node('odom_visualizer_node', anonymous=True)

        self.gt_x, self.gt_y = [], []
        self.pred_x, self.pred_y = [], []
        self.gps_x, self.gps_y = [], []

        self.lock = threading.Lock()

        rospy.Subscriber('/vehicle/odom_ground_truth', Odometry, self.gt_callback)
        rospy.Subscriber('/vehicle/predicted_odom', Odometry, self.pred_callback)
        rospy.Subscriber('/vehicle/gps_odom', Odometry, self.gps_callback)

        self.fig, self.ax = plt.subplots()
        self.gt_line, = self.ax.plot([], [], 'g-', label='Ground Truth')
        self.pred_line, = self.ax.plot([], [], 'r-', label='Predicted')
        self.gps_line, = self.ax.plot([], [], 'b-', label='GPS Odometry')

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_title('Odometry Trajectories')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

    def gt_callback(self, msg):
        with self.lock:
            self.gt_x.append(msg.pose.pose.position.x)
            self.gt_y.append(msg.pose.pose.position.y)

    def pred_callback(self, msg):
        with self.lock:
            self.pred_x.append(msg.pose.pose.position.x)
            self.pred_y.append(msg.pose.pose.position.y)

    def gps_callback(self, msg):
        with self.lock:
            self.gps_x.append(msg.pose.pose.position.x)
            self.gps_y.append(msg.pose.pose.position.y)

    def update_plot(self, frame):
        with self.lock:
            min_len = min(len(self.gt_x), len(self.gt_y), len(self.pred_x), len(self.pred_y), len(self.gps_x), len(self.gps_y))
            if min_len == 0:
                return self.gt_line, self.pred_line, self.gps_line

            self.gt_line.set_data(self.gt_x[:min_len], self.gt_y[:min_len])
            self.pred_line.set_data(self.pred_x[:min_len], self.pred_y[:min_len])
            self.gps_line.set_data(self.gps_x[:min_len], self.gps_y[:min_len])

            all_x = self.gt_x[:min_len] + self.pred_x[:min_len] + self.gps_x[:min_len]
            all_y = self.gt_y[:min_len] + self.pred_y[:min_len] + self.gps_y[:min_len]

            self.ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            self.ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

        return self.gt_line, self.pred_line, self.gps_line

    def start(self):
        # Run ROS spin in a background thread
        ros_thread = threading.Thread(target=rospy.spin)
        ros_thread.daemon = True
        ros_thread.start()

        # Run matplotlib animation in main thread
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()

if __name__ == '__main__':
    visualizer = OdomVisualizer()
    visualizer.start()
