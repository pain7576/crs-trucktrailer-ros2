cmake_minimum_required(VERSION 3.8)
project(wifi_com)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(crs_msgs REQUIRED)
find_package(Protobuf REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${Protobuf_INCLUDE_DIRS})

set(PROTO_FILES
  msgs/proto/CarState.proto
  msgs/proto/IMUMeasurement.proto
  msgs/proto/MotorInput.proto
  msgs/proto/Packet.proto
  msgs/proto/Ping.proto
  msgs/proto/SingleControlInput.proto
  msgs/proto/SteeringPositionMeasurement.proto
  msgs/proto/SteerInput.proto
  msgs/proto/Vector3.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
set_source_files_properties(${PROTO_SRCS} PROPERTIES GENERATED TRUE)

add_executable(wifi_com_node
  src/wifi_com_node.cpp
  src/WiFiCom.cpp
  src/UDPServer.cpp
  ${PROTO_SRCS}
)

ament_target_dependencies(wifi_com_node
  rclcpp
  std_msgs
  sensor_msgs
  crs_msgs
)

target_link_libraries(wifi_com_node
  ${Protobuf_LIBRARIES}
)

install(TARGETS
  wifi_com_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
