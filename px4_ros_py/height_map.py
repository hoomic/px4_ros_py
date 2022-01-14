import rclpy
from rclpy.node import Node
import message_filters

from px4_msgs.msg import *
from std_msgs.msg import *

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from cv_bridge import CvBridge
import cv2

import numpy as np
import quaternion

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.ion()

baseline = 1000

#map size and resolution is in meters
map_size = 200
map_resolution = 1.0

def map_coordinate(x, y):
    x = int((map_size/2 - x) / map_resolution)
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
        self.lidar_sub_ = self.create_subscription(
            LaserScan
            , "scan"
            , self.process_lidar
            , 10
        )
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

        self.init_sub_ = self.create_subscription(
            Bool
            , "height_map/init/in"
            , self.initialize
            , 10
        )

        self.stereo_sub_ = message_filters.TimeSynchronizer(
            [self.left_camera_sub_, self.right_camera_sub_]
            , 10
        )
        self.stereo_sub_.registerCallback(self.process_stereo_images)
        self.br = CvBridge()

        self.mean_publisher_ = self.create_publisher(
            Image
            , "height_map/mean/out"
            , 10
        )

        self.var_publisher_ = self.create_publisher(
            Image
            , "height_map/var/out"
            ,10
        )

        self.metadata_publisher_ = self.create_publisher(
            Float64MultiArray
            , "height_map/metadata/out"
            ,10
        )

        self.resolution_timer_ = self.create_timer(0.5, self.publish_metadata)

        self.initialized = False
        self.last_odom = None
        self.camera_constant = None
        self.mean = np.zeros((int(map_size/map_resolution), int(map_size/map_resolution)))
        self.var = np.zeros((int(map_size/map_resolution), int(map_size/map_resolution))) + 1e6
        self.fig, self.ax = plt.subplots(1, 2, sharex=True, sharey=True)

    def initialize(self, msg):
        self.get_logger().info("Initialized!")
        self.initialized = msg.data

    def publish_metadata(self):
        msg = Float64MultiArray()
        msg.data = [map_resolution, float(map_size)]
        self.metadata_publisher_.publish(msg)

    def process_stereo_images(self, left_image, right_image):
        if not self.initialized:
            return
        #if we haven't received an odometry reading or our camera info, then return
        if self.last_odom is None or self.camera_constant is None:
            return

        # If we are less than 20 meters from the ground, then don't update the map
        if self.last_odom.z > -20:
            mean_msg = self.br.cv2_to_imgmsg(self.mean)
            var_msg = self.br.cv2_to_imgmsg(self.var)
            self.mean_publisher_.publish(mean_msg)
            self.var_publisher_.publish(var_msg)
            
        #convert images into cv2 format
        left_image = self.br.imgmsg_to_cv2(left_image)
        right_image = self.br.imgmsg_to_cv2(right_image)

        #convert images to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        #create rotation quaternion
        rotation = np.quaternion(*self.last_odom.q)

        #create translation quaternion (PX4 uses NED c.s., so z is negated)
        translation = np.quaternion(0, self.last_odom.x, self.last_odom.y, -self.last_odom.z)

        block_sizes = [9]
        update_quaternions = quaternion.as_quat_array(np.zeros((*left_gray.shape, len(block_sizes), 4)))
        for i, bs in enumerate(block_sizes):
            #create stereo block matcher
            stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bs)

            #compute pixel disparities (cv2 returns disparities multiplied by 16, so divide here)
            disparity = stereo.compute(left_gray, right_gray) / 16.0

            #compute X,Y,Z coordinates from disparity map
            X, Y = np.indices(disparity.shape)
            X = np.divide(baseline / 1000 * (X - left_image.shape[1]//2), disparity, out=np.full(disparity.shape, np.nan), where=disparity >0)
            Y = np.divide(baseline / 1000 * (left_image.shape[0]//2 - Y), disparity, out=np.full(disparity.shape, np.nan), where=disparity >0)
            Z = np.divide(self.camera_constant * baseline / 1000.0, -disparity, out=np.zeros(disparity.shape), where=disparity > 0)
            #Z_var = np.zeros(Z.shape) + 5.
            #Z_var = np.clip(winVar(Z, 5), 0, 1e6) + 1

            #perform rigid body transformation of X,Y,Z points in vehicle c.s. to world c.s.
            #convert positions into quaternions
            quat_positions_v = quaternion.as_quat_array(np.dstack([np.zeros(X.shape), X, Y, Z]))
            # rotate vehicle positions into world c.s.
            quat_positions_w = rotation * quat_positions_v * rotation.inverse()

            # translate by vehicle position
            quat_positions_w = quat_positions_w + translation
            update_quaternions[:,:,i] = quat_positions_w
            
        update_quaternions = update_quaternions[np.where(~np.isnan(update_quaternions))]
        self.update(update_quaternions, np.zeros(update_quaternions.shape) + 5.0)
        mean_msg = self.br.cv2_to_imgmsg(self.mean)
        var_msg = self.br.cv2_to_imgmsg(self.var)
        self.mean_publisher_.publish(mean_msg)
        self.var_publisher_.publish(var_msg)

    def process_lidar(self, lidar):
        if not self.initialized:
            return

        if self.last_odom is None:
            return

        #create rotation quaternion
        rotation = np.quaternion(*self.last_odom.q)

        #create translation quaternion (PX4 uses NED c.s., so z is negated)
        translation = np.quaternion(0, self.last_odom.x, self.last_odom.y, -self.last_odom.z)

        angle_inc = lidar.angle_increment
        angle_min = lidar.angle_min
        Y = []
        Z = []
        Z_var = []
        for i, r in enumerate(lidar.ranges):
            theta = angle_min + i * angle_inc
            if np.isinf(r):
                continue
            Y.append(r * np.sin(theta))
            Z.append(r * np.cos(theta))
            Z_var.append(0.02)
        if not len(Y):
            return
        quat_positions_v = quaternion.as_quat_array(np.dstack([np.zeros(len(Y)), np.zeros(len(Y)), Y, Z]))
        # rotate vehicle positions into world c.s.
        quat_positions_w = rotation * quat_positions_v * rotation.inverse()

        # translate by vehicle position
        quat_positions_w = translation - quat_positions_w

        #breakpoint()
        self.update(quat_positions_w.flatten(), np.array(Z_var))
        mean_msg = self.br.cv2_to_imgmsg(self.mean)
        var_msg = self.br.cv2_to_imgmsg(self.var)
        self.mean_publisher_.publish(mean_msg)
        self.var_publisher_.publish(var_msg)

    def process_vehicle_odometry(self, odom):
        self.last_odom = odom

    def process_camera_info(self, info):
        self.camera_constant = info.k[0]

    def update(self, observations, Z_var):
        #observations are 3d points that we measured to exist. For every pixel in a stereo match,
        # we calculate the (x, y, z) coordinate corresponding to that pixel. Thus observations is
        # a list of (x, y, z) coordinates that are occupied. Update must then look up its prediction
        # of the height at (x, y) and update it with the new z measurement
        zplo, zphi = np.percentile([o.z for o in observations], [1, 99])
        m_numer = np.zeros(self.mean.shape)
        v_numer = np.zeros(self.mean.shape)
        denom = np.zeros(self.mean.shape)
        for o, z_var in zip(observations, Z_var):
            if o.z < zplo or o.z > zphi:
                continue
            x, y = map_coordinate(o.x, o.y)
            if 0 <= x < int(map_size/map_resolution) and 0 <= y < int(map_size / map_resolution):
                m_numer[x, y] += o.z
                v_numer[x, y] += z_var
                denom[x, y] += 1
        for x, y in zip(*np.where(denom > 0)):
            m = m_numer[x,y] / denom[x,y]
            v = v_numer[x,y] / denom[x,y]
            self.mean[x,y] = (v * self.mean[x,y] + self.var[x,y] * m) / (v + self.var[x,y])
            self.var[x,y] = 1./(1./self.var[x,y] + 1./v) + 1e-3

    def render(self):
        self.ax[0].imshow(self.mean)
        self.ax[1].imshow(self.var)
        self.fig.canvas.draw()

def main(args=None):
    rclpy.init(args=args)

    height_map = HeightMap()
    
    rclpy.spin(height_map)

    height_map.destroy_node()
    rclpy.shutdown()
