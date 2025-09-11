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
FILE(GLOB proto_files "msgs/proto/*.proto")
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${proto_files})
add_library(${PROJECT_NAME}_proto_msgs
  ${PROTO_SRCS}
)
target_include_directories(${PROJECT_NAME}_proto_msgs PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(${PROJECT_NAME}_proto_msgs
  ${PROTOBUF_LIBRARIES}
)

include_directories(
    include
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(wifi_com_node src/wifi_com_node.cpp src/WiFiCom.cpp src/UDPServer.cpp)
ament_target_dependencies(wifi_com_node
    rclcpp
    std_msgs
    sensor_msgs
    crs_msgs
)
target_link_libraries(wifi_com_node
  ${PROJECT_NAME}_proto_msgs
)

install(TARGETS
  wifi_com_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
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
