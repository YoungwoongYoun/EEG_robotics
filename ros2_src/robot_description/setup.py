from setuptools import setup
from glob import glob
import os

pkg = "robot_description"

setup(
    name=pkg,
    version="0.0.1",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{pkg}"]),
        (f"share/{pkg}", ["package.xml"]),
        (os.path.join("share", pkg, "mesh"), glob("mesh/*")),
        (os.path.join("share", pkg, "urdf"), glob("urdf/*")),
        (os.path.join("share", pkg, "launch"), glob("launch/*.py")),
        (os.path.join("share", pkg, "rviz"), glob("rviz/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ywy",
    maintainer_email="ywy@todo.todo",
    description="Wheelchair robot description (STL + xacro) for RViz TF visualization",
    license="Apache-2.0",
)
