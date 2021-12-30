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

# in mm
baseline = 1000
landing_zone_size = 10000

def quaternion_to_euler(q):
    import math
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.left_camera_sub_ = message_filters.Subscriber(self, Image, 'stereo_camera/left/image_raw')
        self.right_camera_sub_ = message_filters.Subscriber(self, Image, 'stereo_camera/right/image_raw')
        self.vehicle_odometry_sub_ = self.create_subscription(
            VehicleOdometry
            , "fmu/vehicle_odometry/out"
            , self.process_vehicle_odometry
            ,10
        )

        self.camera_info_sub_ = self.create_subscription(
            CameraInfo
            , 'stereo_camera/left/camera_info'
            , self.process_camera_info
            , 10
        )

        self.landing_zone_publisher_ = self.create_publisher(
            TrajectorySetpoint
            , "obstacle_avoidance/landing_zone_adjustment/out"
            , 10
        )

        self.stereo_sub_ = message_filters.TimeSynchronizer(
            [self.left_camera_sub_, self.right_camera_sub_]
            , 10
        )
        self.stereo_sub_.registerCallback(self.process_stereo_images)
        self.br = CvBridge()

        self.destination_reached_subscriber = self.create_subscription(
            Bool
            , "destination_reached/out"
            , self.process_destination_reached
            , 10
        )
        self.destination_reached_ = False
        self.last_odom = None

    def process_destination_reached(self, msg):
        self.destination_reached_ = msg.data
        self.get_logger().info("Destination Reached")

    def process_stereo_images(self, left_image, right_image):
        if self.destination_reached_ and self.last_odom is not None:
            roll, pitch, yaw = quaternion_to_euler(self.last_odom.q)
            if np.abs(roll) < 0.03 and np.abs(pitch) < 0.02:
                self.get_logger().info("Processing stereo images")
                left_frame = self.br.imgmsg_to_cv2(left_image)
                right_frame = self.br.imgmsg_to_cv2(right_image)

                #These are in the coordinate system of the vehicle
                lz_N_v, lz_W_v, z = self.find_landing_zone(left_frame, right_frame, self.last_odom.z)
                x_v = -lz_W_v
                y_v = lz_N_v

                cos_yaw = np.cos(-yaw)
                sin_yaw = np.sin(-yaw)
                #Rotate into world coordinate system
                x_w = x_v * cos_yaw - y_v * sin_yaw
                y_w = x_v * sin_yaw + y_v * cos_yaw

                lz_N = y_w
                lz_E = x_w

                if z == 0 and np.sqrt(lz_N**2 + lz_E **2) < 1.0:
                    z = 5.0

                #publish message to direct vehicle to landing zone
                msg = TrajectorySetpoint()
                msg.x = lz_N
                msg.y = lz_E
                msg.z = z
                self.landing_zone_publisher_.publish(msg)
                self.destination_reached_ = False
            else:
                self.get_logger().info("Destination Reached, but not facing ground roll={} pitch={}".format(roll, pitch))
    
    def process_vehicle_odometry(self, odom):
        self.last_odom = odom

    def process_camera_info(self, info):
        self.camera_constant = info.k[0]

    def find_landing_zone(self, left_image, right_image, z):
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        # calculate the disparity between the left and right images
        disparity = stereo.compute(left_gray, right_gray)
        # count how many pixels have the same disparity
        bin_counts = np.bincount(disparity[np.where(disparity>=0)].flatten())
        # find the modes of this count
        modes = find_modes(bin_counts)
        # filter modes that are less than the 90th percentile 
        p90_mode = np.percentile(modes[:,0], 90)
        min_mode_index = np.max(modes[:,1])
        # the ground should be the mode with the least disparity (ie the farthest away)
        for m in modes:
            if m[0] >= p90_mode and m[1] < min_mode_index:
                min_mode_index = m[1]

        ground_parallax = min_mode_index / 16.0
        disparity = disparity.astype(np.float32) / 16.0
        ground_dist = self.camera_constant * baseline / ground_parallax / 1000.0
        dists = np.divide(self.camera_constant * baseline / 1000.0, disparity, out=np.zeros(disparity.shape), where=disparity > 0)
        abs_diff = np.abs(dists - ground_dist)

        #lz_pixel is the pixel length/width of the desired landing zone
        lz_pixels = int(landing_zone_size * ground_parallax / baseline)
        half_lz = lz_pixels // 2

        #first check if straight down passes the unobstructed test
        lo_N = left_image.shape[0] // 2 - half_lz
        hi_N = left_image.shape[0] // 2 + half_lz
        lo_W = left_image.shape[1] // 2 - half_lz
        hi_W = left_image.shape[1] // 2 + half_lz
        center_lz = abs_diff[lo_N:hi_N, lo_W:hi_W]
        center_p99 = np.percentile(center_lz[np.where(center_lz < ground_dist)], 99)
        breakpoint()
        if center_p99 < 0.5:
            self.get_logger().info("Current landing is unobstructed with 99th percentile dist = {}".format(center_p99))
            return 0.0, 0.0, 0.0

        k = gaussian_kernel(lz_pixels, 1)
        conv = convolve(abs_diff, k)
        conv = conv[lz_pixels:-lz_pixels, lz_pixels:-lz_pixels]
        #scale to have max of 1
        conv = conv/np.max(conv)

        min_val = 1e6
        lz_N, lz_W = 0, 0
        it = np.nditer(conv, flags=['multi_index'])
        while not it.finished:
            N, W = it.multi_index
            c_N = (N - conv.shape[0] // 2) / conv.shape[0]
            c_W = (W - conv.shape[1] // 2) / conv.shape[1]
            val = it[0] + np.sqrt(c_W**2 + c_N**2) * 0.1
            if val < min_val:
                min_val = val
                lz_W = W
                lz_N = N
            it.iternext()
        
        lz_N += lz_pixels
        lz_W += lz_pixels

        p95 = np.percentile(abs_diff[lz_N - half_lz:lz_N + half_lz, lz_W-half_lz:lz_W+half_lz], 95)

        #import matplotlib.pyplot as plt
        #breakpoint()

        if p95 > 5:
            return 0, 0, -10.0

#        left_image[lz_N, lz_W] = [255,255,255]
#        cv2.imshow('left {} {}'.format(lz_W, lz_N), left_image)
#        cv2.waitKey(0)

        lz_N = (left_image.shape[0] // 2) - lz_N
        lz_W = (left_image.shape[1] // 2) - lz_W

        lz_N = baseline * lz_N / ground_parallax / 1000
        lz_W = baseline * lz_W / ground_parallax / 1000

        return lz_N, lz_W, 0.0

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