#ifndef WIFI_COM_WIFICOM_H
#define WIFI_COM_WIFICOM_H

/**
 * @file WiFiCom.h
 * @author Lukas Vogel (vogellu@ethz.ch)
 * @brief Class that communicates with cars over Wi-Fi and UDP.
 */

#pragma once

#include <sys/socket.h>

#include "IMUMeasurement.pb.h"
#include "MotorInput.pb.h"
#include "SingleControlInput.pb.h"
#include "SteeringPositionMeasurement.pb.h"
#include "crs_msgs/msg/car_input.hpp"
#include "rclcpp/rclcpp.hpp"
#include "wifi_com/UDPServer.h"
#include "sensor_msgs/msg/imu.hpp"
#include "crs_msgs/msg/car_ll_control_input.hpp"
#include "crs_msgs/msg/car_steer_state.hpp"



/**
 * @brief Class that communicates with cars over Wi-Fi and UDP.
 *
 * Let ns be the namespace of this node. Then the WiFiCom class subscribes to
 * the /ns/control_input topic and sends the torque and steer commands it
 * receives to the car. The cars send the following data back:
 *
 * - IMU data, which is published on /ns/imu;
 * - the steer position, which is published on /ns/car_steer_state;
 * - the low-level control inputs that the car gave for logging/debugging, which
 *    is published on /ns/car_ll_control_input.
 *
 * The WiFiCom node communicates with the cars via UDP sockets and protocol
 * buffers. To ensure that incoming messages are processed in a timely manner,
 * the socket needs to be polled for new data arriving. This can be achieved by
 * calling the ::poll() method, which should happen at least at twice the rate
 * of incoming packets, so for 250Hz of data from the car, 500Hz polling should
 * be appropriate.
 *
 * Internally, the WiFiCom class uses Google Protocol Buffers for communication
 * with the car. The messages that are sent are defined in ./msgs/proto/ and are
 * compiled into C++ header/source files that deal with the serialization.
 * To add/modify messages, refer to the documentation of the protobuf:
 * https://developers.google.com/protocol-buffers/
 */
class WiFiCom
{
public:
  /**
   * @brief Constructor
   * @param node shared pointer to node
   */
  WiFiCom(std::shared_ptr<rclcpp::Node> node);

  /**
   * @brief Destructor
   */
  ~WiFiCom();

  /**
   * @brief Polls the socket to see if any data was received.
   *
   * The UDP socket does not interrupt the process if new data is received.
   * There need to be periodic checks if new data has arrived, since this node
   * does not run in a multi-threaded manner and can't afford to block on
   * waiting for data.
   */
  void poll();

  /**
   * @brief Callback for incoming /ns/control_input messages.
   *
   * Every /ns/control_input message gets sent to the UDP socket.
   * @param msg the input
   */
  void controllerCallback(const crs_msgs::msg::CarInput::SharedPtr msg);

private:
  /* Private methods -------------------------------------------------------- */
  /** Load the parameters from the configuration file */
  void loadParameters();

  /** Set up all subscriber objects. */
  void setupSubscribers();

  /** Set up all publisher objects. */
  void setupPublishers();

  /** Starts up the UDP server. */
  bool startUDPServer();

  /** Send a control input message from CRS to the car. */
  void sendControlInput(const crs_msgs::msg::CarInput::SharedPtr msg);

  /** Publish the received IMU data on /ns/imu. */
  void publishImuData(const IMUMeasurement& data);

  /** Publish the car's low-level motor inputs on /ns/car_ll_control_input. */
  void publishLowLevelControlInput(const SingleControlInput& reference, const MotorInput& drive_input,
                                   const MotorInput& steer_input);

  /** Publish the car's steerin gpoisition on /ns/car_steer_state */
  void publishSteerState(const SteeringPositionMeasurement& steer_state);

  /* Private member variables ----------------------------------------------- */

  /** Retains the node handle passed to the constructor. */
  std::shared_ptr<rclcpp::Node> node_;

  /** Subscriber object to control_input topic */
  rclcpp::Subscription<crs_msgs::msg::CarInput>::SharedPtr sub_control_input_;

  /** Publisher on the car_ll_control_input topic */
  rclcpp::Publisher<crs_msgs::msg::CarLlControlInput>::SharedPtr pub_car_ll_control_input_;

  /** Publisher on the imu topic */
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;

  /** Publisher on the car_steer_state topic */
  rclcpp::Publisher<crs_msgs::msg::CarSteerState>::SharedPtr pub_car_steer_state_;

  /**
   * UDP server object that handles the sending and receiving of UDP packets at
   * the byte level. Does not know about the protocol buffer layer.
   */
  UDPServer udp_server_;

  /** Port number that the server listens on. */
  int udp_port_;

  /** UDP socket that the node opened */
  int sock_ = -1;

  /** Open a UDP socket according to the configuration. */
  bool openSocket();

  /** Address of the car connecting to the node. */
  struct sockaddr_storage client_;
  socklen_t client_len_;
};
#endif