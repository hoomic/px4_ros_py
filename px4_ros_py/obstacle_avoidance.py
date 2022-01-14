import rclpy
from rclpy.node import Node
import message_filters

from px4_msgs.msg import *
from std_msgs.msg import *

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

import numpy as np
import scipy.stats as st
from scipy.ndimage.filters import convolve

import matplotlib.pyplot as plt

# in mm
baseline = 1000

# in meters
landing_zone_size = 10

def winVar(img, wlen):
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
    borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    return wsqrmean - wmean*wmean



class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.height_map_mean_sub_ = message_filters.Subscriber(self, Image, 'height_map/mean/out')
        self.height_map_var_sub_ = message_filters.Subscriber(self, Image, 'height_map/var/out')
        self.height_map_metadata_sub_ = self.create_subscription(
            Float64MultiArray
            , "height_map/metadata/out"
            , self.set_height_map_metadata
            , 10
        )

        self.trajectory_setpoint_sub_ = self.create_subscription(
            TrajectorySetpoint
            , "fmu/trajectory_setpoint/in"
            , self.process_trajectory_setpoint
            , 10
        )
        self.vehicle_odometry_sub_ = self.create_subscription(
            VehicleOdometry
            , "fmu/vehicle_odometry/out"
            , self.process_vehicle_odometry
            ,10
        )

        self.landing_zone_publisher_ = self.create_publisher(
            TrajectorySetpoint
            , "obstacle_avoidance/landing_zone_adjustment/out"
            , 10
        )

        self.stereo_sub_ = message_filters.TimeSynchronizer(
            [self.height_map_mean_sub_, self.height_map_var_sub_]
            , 10
        )
        self.stereo_sub_.registerCallback(self.process_height_map)
        self.br = CvBridge()

        self.destination_reached_subscriber_ = self.create_subscription(
            Bool
            , "destination_reached/out"
            , self.process_destination_reached
            , 10
        )
        # default behavior is to fly towards initial landing zone, but we can abort if we think it 
        # is occupied. However after we set it to a new desination, then we wait until we reach that
        # destination to set it again
        self.destination_reached_ = True
        self.landing_zone = None
        self.height_map_resolution = None

    def process_destination_reached(self, msg):
        self.destination_reached_ = msg.data
        self.get_logger().info("Destination Reached")

    def process_height_map(self, mean, var):
        if self.height_map_resolution is None or self.landing_zone is None:
            return
        mean = self.br.imgmsg_to_cv2(mean)
        var = self.br.imgmsg_to_cv2(var)
        lz = self.find_landing_zone(mean, var)
        if lz is None:
            return

        lz_N, lz_E, dz = lz

        #publish message to direct vehicle to landing zone
        msg = TrajectorySetpoint()
        msg.x = lz_N
        msg.y = lz_E
        msg.z = dz
        self.landing_zone_publisher_.publish(msg)
        self.destination_reached_ = False

    def process_trajectory_setpoint(self, ts):
        self.get_logger().debug("Setting self.landing_zone=({},{})".format(ts.x, ts.y))
        self.landing_zone = (ts.x, ts.y)

    def process_vehicle_odometry(self, odom):
        self.last_odom = odom

    def set_height_map_metadata(self, metadata):
        self.height_map_resolution = metadata.data[0]
        self.height_map_size = metadata.data[1]

    def find_landing_zone(self, mean, var):
        half_lz = int(landing_zone_size / self.height_map_resolution / 2.)

        lz_var = winVar(mean, int(landing_zone_size / self.height_map_resolution))
        lz_var[:half_lz,:] = 1e6
        lz_var[-half_lz:,:] = 1e6
        lz_var[:,:half_lz] = 1e6
        lz_var[:,-half_lz:] = 1e6
        for i in range(half_lz, mean.shape[0] - half_lz):
            for j in range(half_lz, mean.shape[1] - half_lz):
                var_box = var[i - half_lz:i + half_lz, j - half_lz:j + half_lz]
                lz_var[i, j] = lz_var[i, j] if np.all(var_box < 1e5) else 1e6
        x, y = self.landing_zone
        x, y = self.map_coordinate(x, y)
        if lz_var[x, y] > 1e3:
            self.get_logger().info("Haven't seen landing zone yet, continue...")
            return 
        elif lz_var[x, y] > 0.05 * -self.last_odom.z and self.destination_reached_:
            self.get_logger().info("Landing zone occupied, finding a new one...")
            X, Y = np.indices(lz_var.shape)
            ground_dist = np.abs(mean)
            dist = np.sqrt(np.power((X - x)/(lz_var.shape[0]/2), 2) + np.power((Y - y)/(lz_var.shape[1]/2), 2))
            metric = lz_var + dist * 2.0 + ground_dist * 1.0
            breakpoint()
            new_lz_coords = np.where(metric == np.min(metric))
            new_x = new_lz_coords[0][0]
            new_y = new_lz_coords[1][0]
            x, y = self.world_coordinate(new_x, new_y)
            self.landing_zone = (x, y)
            z_max = np.max(mean)
            dz = (-self.last_odom.z - z_max) / 2.
            return (x, y, dz)

        if self.destination_reached_:
            x, y = self.landing_zone
            return x, y, 5.0
        return
    
    def world_coordinate(self, x, y):
        x = self.height_map_size / 2. - self.height_map_resolution * x
        y = self.height_map_resolution * y - self.height_map_size/2.
        return x, y

    def map_coordinate(self, x, y):
        x = int((self.height_map_size/2 - x) / self.height_map_resolution)
        y = int((y + self.height_map_size/2) / self.height_map_resolution)
        return x, y

def squared_difference(left_image, right_image):
    return np.power(np.subtract(left_image, right_image), 2)

def find_modes(array):
    modes = []
    for i, v in enumerate(array[:-1]):
        if i == 0:
            continue
        if array[i-1] < v and array[i+1] < v:
            modes.append([v, i])
    return np.array(modes)

def gaussian_kernel(kernlen, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def main(args=None):
    rclpy.init(args=args)

    obstacle_avoidance = ObstacleAvoidance()
    
    rclpy.spin(obstacle_avoidance)

    obstacle_avoidance.destroy_node()
    rclpy.shutdown()