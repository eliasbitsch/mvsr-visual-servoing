from setuptools import find_packages, setup

package_name = 'vs_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elias Bitsch',
    maintainer_email='eliasbitsch1@gmail.com',
    description='IBVS controller with hand-eye transform and differential kinematics.',
    license='Apache-2.0',
    entry_points={'console_scripts': ['ibvs_node = vs_control.ibvs_node:main']},
)
