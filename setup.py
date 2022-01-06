from setuptools import setup

package_name = 'px4_ros_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mhooi',
    maintainer_email='michael.r.hooi@gmail.com',
    description='Package to interface ROS2 and PX4 Autopilot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_mission = px4_ros_py.simple_mission:main'
            , 'obstacle_avoidance = px4_ros_py.obstacle_avoidance:main'
            , 'height_map = px4_ros_py.height_map:main'
        ],
    },
)
