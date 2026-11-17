from setuptools import find_packages, setup

package_name = 'vs_perception'

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
    description='Foundation-model target localisation with tracking for visual servoing.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'perception_node = vs_perception.perception_node:main',
            'detect_image = vs_perception.detect_image:main',
            'command_node = vs_perception.command_node:main',
            'vs_cmd = vs_perception.command_node:console',
        ],
    },
)
