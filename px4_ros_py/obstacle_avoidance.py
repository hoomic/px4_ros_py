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
        self.last_ground_dist = 1000.0

    def process_destination_reached(self, msg):
        self.destination_reached_ = msg.data
        self.get_logger().info("Destination Reached")

    def process_stereo_images(self, left_image, right_image):
        if self.destination_reached_ and self.last_odom is not None:

            roll, pitch, yaw = quaternion_to_euler(self.last_odom.q)
            # make sure we are facing the ground
            if np.abs(roll) < 0.03 and np.abs(pitch) < 0.03:
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

                if 0 < z < 5 and np.sqrt(lz_N**2 + lz_E **2) < 1.0:
                    z = 5.0
                self.last_ground_dist -= z

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
        num_disparities = 16 if self.last_ground_dist > 20 else 32
        min_disparities = int(self.camera_constant * baseline / (self.last_ground_dist * 1.1) / 1000.0)
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
        stereo.setMinDisparity(min_disparities)
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        # calculate the disparity between the left and right images
        disparity = stereo.compute(left_gray, right_gray)
        # count how many pixels have the same disparity
        bin_counts = np.bincount(disparity[np.where(disparity >= min_disparities * 16)].flatten())
        #assume that the ground disparity is the first disparity that cumulatively covers 5% of the image
        sum = 0
        for i, b in enumerate(bin_counts):
            sum += b
            if sum / np.sum(bin_counts) > 0.05:
                ground_disparity = i
                break

        ground_parallax = ground_disparity / 16.0
        disparity = disparity.astype(np.float32) / 16.0
        ground_dist = self.camera_constant * baseline / ground_parallax / 1000.0
        self.last_ground_dist = ground_dist
        dists = np.divide(self.camera_constant * baseline / 1000.0, disparity, out=np.zeros(disparity.shape), where=disparity >= ground_parallax-1)
        abs_diff = np.abs(dists - ground_dist)

        #lz_pixel is the pixel length/width of the desired landing zone
        lz_pixels = int(landing_zone_size * ground_parallax / baseline)
        half_lz = lz_pixels // 2

        # check to see if the landing zone your are currerntly centered over is clear
        Nlo = abs_diff.shape[0]//2 - half_lz
        Nhi = Nlo + lz_pixels
        Wlo = abs_diff.shape[1]//2 - half_lz
        Whi = Wlo + lz_pixels
        center_lz = abs_diff[Nlo:Nhi, Wlo:Whi]
        if np.all(center_lz[np.where(center_lz < ground_dist)] < ground_dist * 0.05):
            self.get_logger().info("Current landing is clear!")
            return 0.0, 0.0, ground_dist * 0.2

        # convolve the absolute difference with a kernel that is the size of the landing zone
        k = np.zeros((lz_pixels, lz_pixels)) + 1./lz_pixels**2
        conv = convolve(abs_diff, k)
        conv = conv[half_lz:-half_lz, half_lz:-half_lz]

        min_val = 1e6
        lz_N, lz_W = 0, 0
        it = np.nditer(conv, flags=['multi_index'])
        while not it.finished:
            N, W = it.multi_index
            c_N = (N - conv.shape[0] // 2) / conv.shape[0]
            c_W = (W - conv.shape[1] // 2) / conv.shape[1]
            val = it[0] + np.sqrt(c_W**2 + c_N**2) * 0.5
            if val < min_val:
                min_val = val
                lz_W = W
                lz_N = N
            it.iternext()
        
        lz_N += half_lz
        lz_W += half_lz

        p95 = np.percentile(abs_diff[lz_N - half_lz:lz_N + half_lz, lz_W-half_lz:lz_W+half_lz], 95)

        import matplotlib.pyplot as plt
        breakpoint()

        if p95 > ground_dist * 0.2:
            return 0, 0, -10.0

#        left_image[lz_N, lz_W] = [255,255,255]
#        cv2.imshow('left {} {}'.format(lz_W, lz_N), left_image)
#        cv2.waitKey(0)

        lz_N = (left_image.shape[0] // 2) - lz_N
        lz_W = (left_image.shape[1] // 2) - lz_W

        lz_N = baseline * lz_N / ground_parallax / 1000
        lz_W = baseline * lz_W / ground_parallax / 1000

        return lz_N, lz_W, ground_dist * 0.2

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