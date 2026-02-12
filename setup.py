from setuptools import find_packages, setup

package_name = 'hazmap'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        ('share/' + package_name + '/launch', [
            'launch/hazmap.launch.py',
        ]),

        ('share/' + package_name + '/config', [
            'config/hazmap_params.yaml',
            'config/nav2_params.yaml',
            'config/hazmap.rviz',
        ]),
    ],
    install_requires=['setuptools', 'matplotlib'],
    zip_safe=True,
    maintainer='gk',
    maintainer_email='adityakaleeswargk04@gmail.com',
    description='HazMap — Hazard Mapping & Coverage Path Planning for TurtleBot3 (ROS2 Humble)',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'hazmap_node = hazmap.hazmap_core.hazmap_node:main',
        ],
    },
)
