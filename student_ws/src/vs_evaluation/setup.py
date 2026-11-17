from setuptools import setup

package_name = 'vs_evaluation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elias Bitsch',
    maintainer_email='eliasbitsch1@gmail.com',
    description='Automated evaluation runs over targets and start configurations.',
    license='Apache-2.0',
    entry_points={'console_scripts': ['run_eval = vs_evaluation.run_eval:main']},
)
