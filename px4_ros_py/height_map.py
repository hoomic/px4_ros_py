import rclpy
from rclpy.node import Node
import message_filters

from px4_msgs.msg import *
from std_msgs.msg import *

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

import numpy as np
import quaternion

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

baseline = 1000

class HeightMap(Node):
    def __init__(self):
        super().__init__('height_map')

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

        self.last_odom = None

    def process_destination_reached(self, msg):
        self.destination_reached_ = msg.data
        self.get_logger().info("Destination Reached")

    def process_stereo_images(self, left_image, right_image):
        if self.last_odom is not None:
            #convert images into cv2 format
            left_image = self.br.imgmsg_to_cv2(left_image)
            right_image = self.br.imgmsg_to_cv2(right_image)

            #create stereo block matcher
            stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

            #convert images to grayscale
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

            #compute pixel disparities (cv2 returns disparities multiplied by 16, so divide here)
            disparity = stereo.compute(left_gray, right_gray) / 16.0

            #compute X,Y,Z coordinates from disparity map
            Y, X = np.indices(disparity.shape)
            X = np.divide(baseline / 1000 * (X - left_image.shape[1]//2), disparity, out=np.full(disparity.shape, np.nan), where=disparity >0)
            Y = np.divide(baseline / 1000 * (left_image.shape[0]//2 - Y), disparity, out=np.full(disparity.shape, np.nan), where=disparity >0)
            Z = np.divide(self.camera_constant * baseline / 1000.0, disparity, out=np.zeros(disparity.shape), where=disparity > 0)

            #perform rigid body transformation of X,Y,Z points in vehicle c.s. to world c.s.
            #convert positions into quaternions
            quat_positions_v = quaternion.as_quat_array(np.dstack([np.zeros(X.shape), X, Y, Z]))
            #create rotation quaternion
            rotation = np.quaternion(*self.last_odom.q)
            # rotate vehicle positions into world c.s.
            quat_positions_w = rotation * quat_positions_v * rotation.inverse()
            #create translation quaternion
            translation = np.quaternion(0, self.last_odom.x, self.last_odom.y, self.last_odom.z)
            breakpoint()


    def process_vehicle_odometry(self, odom):
        self.last_odom = odom

    def process_camera_info(self, info):
        self.camera_constant = info.k[0]

    def update(self, observations):
        #observations are 3d points that we measured to exist. For every pixel in a stereo match,
        # we calculate the (x, y, z) coordinate correspoinding to that pixel. Thus observations is
        # a list of (x, y, z) coordinates that are occupied. Update must then look up its prediction
        # of the height at (x, y) and update it with the new z measurement
        pass

def main(args=None):
    rclpy.init(args=args)

    height_map = HeightMap()
    
    rclpy.spin(height_map)

    height_map.destroy_node()
    rclpy.shutdown()