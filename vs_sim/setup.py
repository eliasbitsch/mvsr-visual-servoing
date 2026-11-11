import os
from glob import glob

from setuptools import setup

package_name = "vs_sim"


def tree(src):
    """data_files entries for a directory tree (keeps the folder structure)."""
    out = {}
    for path in glob(os.path.join(src, "**", "*"), recursive=True):
        if os.path.isfile(path):
            dest = os.path.join("share", package_name, os.path.dirname(path))
            out.setdefault(dest, []).append(path)
    return list(out.items())


setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *tree("launch"),
        *tree("urdf"),
        *tree("config"),
        *tree("worlds"),
        *tree("models"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="David Seyser",
    maintainer_email="david.t.seyser@gmail.com",
    description="UR5e with wrist camera in Gazebo Harmonic for the visual servoing project",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ground_truth = vs_sim.ground_truth:main",
            "go_to_start = vs_sim.go_to_start:main",
            "capture_reference = vs_sim.capture_reference:main",
            "check_start_configs = vs_sim.check_start_configs:main",
            "example_joint_velocity = vs_sim.example_joint_velocity:main",
        ],
    },
)
