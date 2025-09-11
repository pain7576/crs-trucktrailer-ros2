#include "rclcpp/rclcpp.hpp"
#include "wifi_com/WiFiCom.h"
#include <memory>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto wifi_com_node = std::make_shared<WiFiCom>();
  rclcpp::spin(wifi_com_node);
  rclcpp::shutdown();
  return 0;
}