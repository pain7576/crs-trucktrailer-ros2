#ifndef WIFI_COM_WIFICOM_H
#define WIFI_COM_WIFICOM_H

#include "rclcpp/rclcpp.hpp"
#include "crs_msgs/msg/car_input.hpp"
#include "crs_msgs/msg/car_ll_control_input.hpp"
#include "crs_msgs/msg/car_steer_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "wifi_com/UDPServer.h"
#include <sys/socket.h>
#include "Packet.pb.h"

// Other necessary includes

class WiFiCom : public rclcpp::Node
{
public:
  WiFiCom();
  ~WiFiCom();

private:
  void poll();
  void controllerCallback(const crs_msgs::msg::CarInput::SharedPtr msg);

  void loadParameters();
  void setupSubscribers();
  void setupPublishers();
  bool startUDPServer();

  void sendControlInput(const crs_msgs::msg::CarInput::SharedPtr msg);
  void publishImuData(const IMUMeasurement& data);
  void publishLowLevelControlInput(const SingleControlInput& reference, const MotorInput& drive_input,
                                   const MotorInput& steer_input);
  void publishSteerState(const SteeringPositionMeasurement& steer_state);

  rclcpp::Subscription<crs_msgs::msg::CarInput>::SharedPtr sub_control_input_;
  rclcpp::Publisher<crs_msgs::msg::CarLlControlInput>::SharedPtr pub_car_ll_control_input_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
  rclcpp::Publisher<crs_msgs::msg::CarSteerState>::SharedPtr pub_car_steer_state_;

  UDPServer udp_server_;
  int udp_port_;
  struct sockaddr_storage client_;

  rclcpp::TimerBase::SharedPtr timer_;
};

#endif // WIFI_COM_WIFICOM_H