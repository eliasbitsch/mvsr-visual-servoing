# Stages:
#   sim : ROS 2 Jazzy, Gazebo Harmonic, ros2_control, MoveIt 2, RealSense, Robotiq, pinocchio
#   fm  : sim + CUDA PyTorch and foundation model libraries (default target)
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
      python3-pip python3-colcon-common-extensions python3-rosdep \
      mesa-utils git wget nano \
    && rm -rf /var/lib/apt/lists/*

# PEP 668: inside the container we install into the system interpreter so ROS nodes see the
# packages. numpy<2 keeps the apt-built ROS Python extensions working.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1
COPY docker/constraints.txt /opt/constraints.txt
# The spec requires OpenCV 4.12; the apt OpenCV (4.6) stays for the C++ ROS packages.
RUN pip install -c /opt/constraints.txt opencv-contrib-python-headless==4.12.0.88 scipy matplotlib pandas

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
RUN pip install -c /opt/constraints.txt \
      torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install -c /opt/constraints.txt \
      "transformers>=4.51" accelerate timm einops \
      ultralytics \
      "git+https://github.com/ultralytics/CLIP.git" \
    && { pip uninstall -y opencv-python opencv-python-headless || true; } \
    && pip install --force-reinstall --no-deps opencv-contrib-python-headless==4.12.0.88

# Model weights are cached here; compose mounts a named volume so they survive rebuilds.
ENV HF_HOME=/models/huggingface \
    YOLO_CONFIG_DIR=/models/ultralytics \
    VS_WEIGHTS_DIR=/models/ultralytics/weights \
    TORCH_HOME=/models/torch
