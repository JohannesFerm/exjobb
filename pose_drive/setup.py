from setuptools import setup
import os
from glob import glob

package_name = "pose_drive"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "torch"],
    zip_safe=True,
    description="TESTING",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pose_drive = pose_drive.pose_drive:main"
        ],
    },
)

