import rclpy
from rclpy.node import Node

from px4_msgs.msg import *
from std_msgs.msg import *

import numpy as np

EARTH_RADIUS = 6371000.0
lat_desired = 37.414025
lon_desired = -121.999340
takeoff_altitude = -30.0
cruise_altitude = -100.0
loiter_altitude = -30.0

class SimpleMission(Node):
    def __init__(self):
        super().__init__('simple_mission')
        self.offboard_control_mode_publisher_ = self.create_publisher(
            OffboardControlMode
            , "fmu/offboard_control_mode/in"
            , 10
        )
        self.trajectory_setpoint_publisher_ = self.create_publisher(
            TrajectorySetpoint
            , "fmu/trajectory_setpoint/in"
            , 10
        )
        self.vehicle_command_publisher_ = self.create_publisher(
            VehicleCommand
            , "fmu/vehicle_command/in"
            , 10
        )
        self.destination_reached_publisher_ = self.create_publisher(
            Bool
            , "destination_reached/out"
            , 10
        )
        self.height_map_init_publisher_ = self.create_publisher(
            Bool
            , "height_map/init/in"
            , 10
        )

        self.vehicle_odometry_subscriber_ = self.create_subscription(
            VehicleOdometry
            , "fmu/vehicle_odometry/out"
            , self.process_vehicle_odometry
            , 10
        )
        self.vehicle_gps_position_subscriber_ = self.create_subscription(
            VehicleGpsPosition
            , "fmu/vehicle_gps_position/out"
            , self.process_vehicle_gps_position
            , 10
        )
        self.vehicle_status_subscriber_ = self.create_subscription(
            VehicleStatus
            , "fmu/vehicle_status/out"
            , self.process_vehicle_status
            , 10
        )
        self.landing_zone_subscriber_ = self.create_subscription(
            TrajectorySetpoint
            , "obstacle_avoidance/landing_zone_adjustment/out"
            , self.process_landing_zone_adjustment
            , 10
        )
        self.height_map_metadata_sub_ = self.create_subscription(
            Float64MultiArray
            , "height_map/metadata/out"
            , self.set_height_map_metadata
            , 10
        )
        self.timesync_sub_ = self.create_subscription(
            Timesync
            , "fmu/timesync/out"
            , self.process_timesync
            , 10
        )

        self.timestamp_ = 0
        self.offboard_setpoint_counter_ = 0
        self.takeoff_complete_ = False
        self.back_transitioned_ = False
        self.destination_reached_ = False
        self.waypoint_initialized_ = False

        self.initial_altitude_ = 0.0
        self.lz_x = 0.0
        self.lz_y = 0.0
        self.target_yaw_ = 0.0
        self.target_altitude_ = takeoff_altitude

        self.height_map_resolution = 0
        self.height_map_size = 0
        self.height_map_initiated = False

    def process_timesync(self, msg):
        self.timestamp_ = msg.timestamp

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

        self.get_logger().info("Arm command send")

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)

        self.get_logger().info("Disarm command send")

    def process_vehicle_status(self, status):
        if not self.destination_reached_:
            if status.arming_state == 1 and self.offboard_setpoint_counter_ > 10:
                self.set_gps_global_origin()
                self.get_logger().info("Switching to Offboard Mode")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

                self.arm()

    def process_vehicle_odometry(self, odom):
        self.publish_offboard_control_mode()
        if self.destination_reached_ or not self.waypoint_initialized_:
            return

        msg = TrajectorySetpoint()
        msg.timestamp = self.timestamp_
        msg.yaw = self.target_yaw_
        dist_to_dest = np.sqrt((odom.x - self.lz_x)**2 + (odom.y - self.lz_y)**2 + (odom.z - self.target_altitude_)**2)

        if not self.takeoff_complete_:
            msg.x = odom.x
            msg.y = odom.y
            if np.abs(odom.z - takeoff_altitude) < 1.0:
                self.transition_to_fw()
                self.takeoff_complete_ = True
                self.target_altitude_ = cruise_altitude
        else:
            if dist_to_dest < 0.1:
                self.get_logger().info("Destination Reached")
                self.destination_reached_ = True
                msg = Bool()
                msg.data = True
                self.destination_reached_publisher_.publish(msg)
                return
            if dist_to_dest < self.height_map_size and not self.height_map_initiated:
                self.height_map_initiated = True
                init_msg = Bool()
                init_msg.data = True
                self.height_map_init_publisher_.publish(init_msg)
            if dist_to_dest < 100:
                if not self.back_transitioned_:
                    self.get_logger().info("Approaching Destination fly down to landing altitude")
                    self.transition_to_mc()
                    self.back_transitioned_ = True
                    self.target_altitude_ = loiter_altitude
            msg.x = self.lz_x
            msg.y = self.lz_y
        msg.z = self.target_altitude_
        self.trajectory_setpoint_publisher_.publish(msg)
        self.offboard_setpoint_counter_ += 1

    def process_landing_zone_adjustment(self, lza):
        self.get_logger().info("Process Landing Zone Adjustment x={0:.2f} y={1:.2f} dz={2:.2f}".format(lza.x, lza.y, lza.z))
        self.target_altitude_ += lza.z
        if np.sqrt((lza.x - self.lz_x)**2 + (lza.y - self.lz_y)**2) > 1.0 or self.target_altitude_ <= -10:
            self.destination_reached_ = False
        else:
            self.get_logger().info("Landing!")
            self.auto_land()
        self.lz_x = lza.x
        self.lz_y = lza.y

    def set_height_map_metadata(self, metadata):
        self.height_map_resolution = metadata.data[0]
        self.height_map_size = metadata.data[1]

    def process_vehicle_gps_position(self, gps):
        if not self.waypoint_initialized_:
            self.initial_altitude_ = gps.alt / 1000.0
            curr_lat = np.deg2rad(gps.lat * 1e-7)
            curr_lon = np.deg2rad(gps.lon * 1e-7)
            delta_x = EARTH_RADIUS * (np.deg2rad(lon_desired) - curr_lon) * np.cos(curr_lat)
            delta_y = EARTH_RADIUS * (np.deg2rad(lat_desired) - curr_lat)
            self.target_yaw_ = -np.arctan2(delta_y, delta_x) + np.pi/2
            self.waypoint_initialized_ = True

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.timestamp_
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False

        self.offboard_control_mode_publisher_.publish(msg)

    def transition_to_fw(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_VTOL_TRANSITION, 4.0, 1.0)
        self.get_logger().info("Transition to FW command send")

    def transition_to_mc(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_VTOL_TRANSITION, 3.0, 1.0)
        self.get_logger().info("Transition to MC command send")

    def publish_vehicle_command(self, command, param1 = 0.0, param2 = 0.0):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp_
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.vehicle_command_publisher_.publish(msg)

    def set_gps_global_origin(self):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp_
        msg.param5 = lat_desired
        msg.param6 = lon_desired
        msg.command = VehicleCommand.VEHICLE_CMD_SET_GPS_GLOBAL_ORIGIN
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.vehicle_command_publisher_.publish(msg)

    def auto_land(self):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp_
        msg.param1 = 1.0
        msg.param2 = 4.0
        msg.param3 = 6.0
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.vehicle_command_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    simple_mission = SimpleMission()
    
    rclpy.spin(simple_mission)

    simple_mission.destroy_node()
    rclpy.shutdown()