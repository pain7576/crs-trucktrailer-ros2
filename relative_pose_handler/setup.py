from setuptools import find_packages, setup

package_name = 'relative_pose_handler'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='harsh',
    maintainer_email='harsh@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'natnet_sim = relative_pose_handler.pub_natnet_sim:main',
            'angle_finder = relative_pose_handler.world_2_angle:main',
            'control_sim = relative_pose_handler.pub_controller_sim:main',
            'car_pose = relative_pose_handler.pub_car_world_2_my_world:main',
            'pose = relative_pose_handler.car_publisher:main',
            'sys = relative_pose_handler.trailer_publisher:main',
            'render = relative_pose_handler.truck_trailer_publisher:main',
        ],
    },
)
