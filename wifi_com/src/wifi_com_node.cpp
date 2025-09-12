/**
 * @file    wifi_com_node.cpp
 * @author  Lukas Vogel
 * @brief   Node that instantiates the WifiCom object.
 */
#include "rclcpp/rclcpp.hpp"
#include "crs_msgs/msg/car_com.hpp"
#include "std_msgs/msg/string.hpp"
#include "wifi_com/WiFiCom.h"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("wifi_com");

  // The wifi_com_node should run at 500Hz, double the rate that packets will
  // arrive. This way, incoming packets should not suffer from additional large
  // delays (=> 1-2ms added by the polling period)
  rclcpp::Rate rate(500);
  WiFiCom com(node);

  while (rclcpp::ok())
  {
    // process all incoming messages first, this will send outgoing packets
    rclcpp::spin_some(node);
    // poll socket for incoming packets and publish the data as message
    com.poll();
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}