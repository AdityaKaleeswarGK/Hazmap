from setuptools import find_packages, setup

package_name = 'prometheus_frontier_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        ('share/' + package_name + '/launch', [
            'launch/nav2_bringup.launch.py',
            'launch/gazebo_rviz.launch.py',
            'launch/slam.launch.py',
            'launch/frontier_exploration.launch.py'
        ]),

        ('share/' + package_name + '/worlds', [
            'world/prometheus_world.world'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gk',
    maintainer_email='adityakaleeswargk04@gmail.com',
    description='Frontier exploration package using Prometheus algorithm in ROS 2',
    license='BSD-3-Clause',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'frontier_explorer = prometheus_frontier_explorer.frontier_explorer:main',
            'frontier_visualizer = prometheus_frontier_explorer.map:main',
        ],
    },
)
