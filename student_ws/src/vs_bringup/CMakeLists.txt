cmake_minimum_required(VERSION 3.8)
project(vs_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/viz_relay.py DESTINATION lib/${PROJECT_NAME})

ament_package()
