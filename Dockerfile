# UR5e visual servoing simulation: ROS 2 Jazzy + Gazebo Harmonic
# ros:jazzy-ros-base is multi-arch (amd64 + arm64, e.g. Apple Silicon)
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ur-simulation-gz \
        ros-jazzy-ur-description \
        ros-jazzy-ros-gz \
        ros-jazzy-gz-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-controller-manager \
        ros-jazzy-ros2controlcli \
        ros-jazzy-xacro \
        ros-jazzy-cv-bridge \
        ros-jazzy-rqt-image-view \
        ros-jazzy-rviz2 \
        ros-jazzy-tf2-ros-py \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libegl1 \
        libegl-mesa0 \
        python3-opencv \
        python3-numpy \
        python3-scipy \
        python3-yaml \
        python3-pytest \
        python3-pip \
        python3-colcon-common-extensions \
        mesa-utils \
        nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY vs_sim /ws/src/vs_sim
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install --packages-select vs_sim

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/headless_test.sh /headless_test.sh
RUN chmod +x /entrypoint.sh /headless_test.sh \
    && echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
    && echo "source /ws/install/setup.bash" >> /root/.bashrc \
    && echo "[ -f /ws/student/install/setup.bash ] && source /ws/student/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
