# Stages:
#   sim : ROS 2 Jazzy, Gazebo Harmonic, ros2_control, MoveIt 2, RealSense, Robotiq, pinocchio
#   fm  : sim + CUDA PyTorch and foundation model libraries (default target)
#
# Two Python environments, both Python 3.12:
#   system  /usr/bin/python3     numpy 1.26 (apt). ROS, pinocchio -> vs_control
#   venv    /opt/vs-venv         numpy 2, OpenCV 4.12, PyTorch -> vs_perception
# OpenCV 4.12 (required by the spec) needs numpy 2, and the apt pinocchio segfaults with numpy 2.
FROM osrf/ros:jazzy-desktop-full AS sim

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# upgrade first: the base image is older than the repo, and mixed ros2_control versions make
# gz_ros2_control crash when the robot is loaded (seen in AROBL)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
      libgl1-mesa-dri libegl1 libegl-mesa0 libglx-mesa0 \
      ros-jazzy-ros-gz \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-moveit \
      ros-jazzy-moveit-servo \
      ros-jazzy-moveit-visual-tools \
      ros-jazzy-realsense2-camera \
      ros-jazzy-realsense2-description \
      ros-jazzy-robotiq-description \
      ros-jazzy-pinocchio \
      ros-jazzy-tf-transformations \
      ros-jazzy-rqt-image-view \
      ros-jazzy-plotjuggler-ros \
      python3-pip python3-venv python3-colcon-common-extensions python3-rosdep \
      mesa-utils git wget nano \
    && rm -rf /var/lib/apt/lists/*

# The pip cache lives in a BuildKit cache mount (not in the image): after a dropped download a
# rebuild only fetches what is missing, which matters for the 2.5 GB of PyTorch wheels.
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    VS_PYTHON=/opt/vs-venv/bin/python
COPY docker/constraints.txt /opt/constraints.txt
# --system-site-packages: rclpy and the generated messages come from /opt/ros
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv --system-site-packages /opt/vs-venv \
    && /opt/vs-venv/bin/pip install -c /opt/constraints.txt \
         numpy opencv-contrib-python-headless scipy matplotlib pandas

ENV ROS_WS=/ros2_ws
WORKDIR ${ROS_WS}
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc \
    && echo '[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash' >> /root/.bashrc
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]


FROM sim AS fm

# CUDA 12.4 wheels bundle their own CUDA runtime, only the host driver is needed.
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/vs-venv/bin/pip install -c /opt/constraints.txt \
      torch torchvision --index-url https://download.pytorch.org/whl/cu124
# ultralytics pulls opencv-python, which would shadow the contrib build (CSRT/KCF trackers)
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/vs-venv/bin/pip install -c /opt/constraints.txt \
      "transformers>=4.51" accelerate timm einops \
      ultralytics \
      "git+https://github.com/ultralytics/CLIP.git" \
    && { /opt/vs-venv/bin/pip uninstall -y opencv-python opencv-python-headless || true; } \
    && /opt/vs-venv/bin/pip install --force-reinstall --no-deps -c /opt/constraints.txt \
         opencv-contrib-python-headless

# Model weights are cached here; compose mounts a named volume so they survive rebuilds.
ENV HF_HOME=/models/huggingface \
    YOLO_CONFIG_DIR=/models/ultralytics \
    VS_WEIGHTS_DIR=/models/ultralytics/weights \
    TORCH_HOME=/models/torch
