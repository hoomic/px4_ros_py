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

plt.ion()

baseline = 1000

#map size and resolution is in meters
map_size = 400
map_resolution = 0.5

def map_coordinate(x, y):
    x = int((x + map_size/2) / map_resolution)
    y = int((y + map_size/2) / map_resolution)
    return x, y

def winVar(img, wlen):
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
    borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    return wsqrmean - wmean*wmean

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
        self.camera_constant = None
        self.height_map_mean = np.zeros((int(map_size/map_resolution), int(map_size/map_resolution)))
        self.height_map_var = np.zeros((int(map_size/map_resolution), int(map_size/map_resolution))) + 1e6
        self.fig, self.ax = plt.subplots(1, 2, sharex=True, sharey=True)

    def process_destination_reached(self, msg):
        self.destination_reached_ = msg.data
        self.get_logger().info("Destination Reached")
        self.render()
        breakpoint()

    def process_stereo_images(self, left_image, right_image):
        if self.last_odom is not None and self.camera_constant is not None:
            if self.last_odom.z < -20:
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
                Z = np.divide(self.camera_constant * baseline / 1000.0, -disparity, out=np.zeros(disparity.shape), where=disparity > 0)
                Z_var = np.clip(winVar(Z, 5), 0, 1e6) + 1

                #perform rigid body transformation of X,Y,Z points in vehicle c.s. to world c.s.
                #convert positions into quaternions
                quat_positions_v = quaternion.as_quat_array(np.dstack([np.zeros(X.shape), X, Y, Z]))
                #create rotation quaternion
                rotation = np.quaternion(*self.last_odom.q)
                # rotate vehicle positions into world c.s.
                quat_positions_w = rotation * quat_positions_v * rotation.inverse()
                #create translation quaternion (PX4 uses NED c.s., so x and y are switched and z is negated)
                translation = np.quaternion(0, self.last_odom.y, self.last_odom.x, -self.last_odom.z)
                # translate by vehicle position
                quat_positions_w = quat_positions_w + translation
                self.update(quat_positions_w[np.where(~np.isnan(quat_positions_w))]
                            , Z_var[np.where(~ np.isnan(quat_positions_w))])

    def process_vehicle_odometry(self, odom):
        self.last_odom = odom

    def process_camera_info(self, info):
        self.camera_constant = info.k[0]

    def update(self, observations, Z_var):
        #observations are 3d points that we measured to exist. For every pixel in a stereo match,
        # we calculate the (x, y, z) coordinate corresponding to that pixel. Thus observations is
        # a list of (x, y, z) coordinates that are occupied. Update must then look up its prediction
        # of the height at (x, y) and update it with the new z measurement
        zp5, zp95 = np.percentile([o.z for o in observations], [5,95])
        for o, z_var in zip(observations, Z_var):
            if o.z < zp5 or o.z > zp95:
                continue
            x, y = map_coordinate(o.x, o.y)
            if 0 <= x < int(map_size/map_resolution) and 0 <= y < int(map_size / map_resolution):
                self.height_map_mean[x,y] = (z_var * self.height_map_mean[x,y] + self.height_map_var[x,y] * o.z) / (z_var + self.height_map_var[x,y])
                self.height_map_var[x,y] = 1./(1./self.height_map_var[x,y] + 1./z_var) + 1e-3

    def render(self):
        self.ax[0].imshow(self.height_map_mean)
        self.ax[1].imshow(self.height_map_var)
        self.fig.canvas.draw()

def main(args=None):
    rclpy.init(args=args)

    height_map = HeightMap()
    
    rclpy.spin(height_map)

    height_map.destroy_node()
    rclpy.shutdown()