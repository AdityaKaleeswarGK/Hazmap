import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'app'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.*'))),
        (os.path.join('share', package_name, 'models'), glob(os.path.join('models', '*.*'))),
    ],
    install_requires=['setuptools', 'smbus2'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='1270161395@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_controller = app.lidar_controller:main',
            'line_following = app.line_following:main',
            'object_tracking = app.object_tracking:main',
            'ar_app = app.ar_app:main',
            'patrol = app.patrol:main',
            'gas_field_sim = app.gas_field_sim:main',
            'gas_field_sim_multi = app.gas_field_sim_multi:main',
            'mq_ads1115_node = app.mq_ads1115_node:main',
            'mq_ads1115_multi = app.mq_ads1115_multi:main',
            'gas_mapper = app.gas_mapper:main',
        ],
    },
)
