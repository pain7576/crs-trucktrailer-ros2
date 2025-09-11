/**
 * @file WiFiCom.cpp
 * @author Lukas Vogel (vogellu@ethz.ch)
 * @brief Class that communicates with cars over Wi-Fi and UDP.
 * @brief ROS 2 Migration by Gemini
 */

#include "wifi_com/WiFiCom.h"

#include <arpa/inet.h>
#include <boost/interprocess/streams/bufferstream.hpp>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <string>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>

#include "Packet.pb.h"

using namespace std::chrono_literals;

/** Default port number to open the UDP server on in case nothing was specified
    in the config file. */
#define DEFAULT_PORT_NUM 20211

/** Maximum recommended size of a UDP packet that is sent without complaining. */
#define UDP_MAX_RECOMMENDED_SIZE 256
/** Gravitational acceleration, used to convert IMU data from g's to m/s^2 */
#define GRAVITATIONAL_ACCELERATION 9.81

/* Public method implementation --------------------------------------------- */

WiFiCom::WiFiCom() : rclcpp::Node("wifi_com"), udp_port_(DEFAULT_PORT_NUM)
{
  loadParameters();
  setupPublishers();
  setupSubscribers();
  startUDPServer();

  // The original node ran at 500Hz. We create a wall timer to call poll() at the same rate.
  timer_ = this->create_wall_timer(2ms, std::bind(&WiFiCom::poll, this));
}

WiFiCom::~WiFiCom()
{
  udp_server_.stopListening();
}

void WiFiCom::poll()
{
  uint8_t rx_buffer[512];
  ssize_t recv_len = udp_server_.pollReceive(&client_, rx_buffer, sizeof(rx_buffer));

  if (recv_len < 0)
  {
    RCLCPP_ERROR(this->get_logger(), "Polling of UDP server threw an error! Code: %ld", recv_len);
    return;
  }
  if (recv_len == 0)
  {
    // no new data arrived
    return;
  }

  // Byte array is null terminated, so can cast to string
  boost::interprocess::bufferstream rx_istream((char*)rx_buffer, recv_len);

  Packet p;
  if (!p.ParseFromIstream(&rx_istream))
  {
    RCLCPP_ERROR(this->get_logger(), "Could not parse packet from input stream.");
    return;
  }

  if (p.has_car_state())
  {
    const CarState& state = p.car_state();
    if (state.has_drive_motor_input() && state.has_steer_motor_input() && state.has_current_reference())
    {
      publishLowLevelControlInput(state.current_reference(), state.drive_motor_input(), state.steer_motor_input());
    }

    if (state.has_imu_data())
    {
      publishImuData(state.imu_data());
    }

    if (state.has_steer_data())
    {
      publishSteerState(state.steer_data());
    }
  }
  else
  {
    RCLCPP_WARN(this->get_logger(), "Received unknown packet type, dropping...");
  }
}

void WiFiCom::controllerCallback(const crs_msgs::msg::CarInput::SharedPtr msg)
{
  sendControlInput(msg);
}

/* Private method implementation -------------------------------------------- */

void WiFiCom::loadParameters()
{
  RCLCPP_INFO(this->get_logger(), "WiFiCom: loading parameters");
  this->declare_parameter<int>("udp_port", DEFAULT_PORT_NUM);
  this->get_parameter("udp_port", udp_port_);
  RCLCPP_INFO(this->get_logger(), "Using UDP port: %d", udp_port_);
}

void WiFiCom::setupSubscribers()
{
  // In ROS 2, QoS settings are important. 10 is a common history depth.
  sub_control_input_ = this->create_subscription<crs_msgs::msg::CarInput>(
      "control_input", 10, std::bind(&WiFiCom::controllerCallback, this, std::placeholders::_1));
}

void WiFiCom::setupPublishers()
{
  // While all these messages might arrive in the same packet from the car, each
  // belongs to a separate CRS topic and they are thus published on those:
  pub_car_ll_control_input_ = this->create_publisher<crs_msgs::msg::CarLlControlInput>("car_ll_control_input", 10);
  pub_imu_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);
  pub_car_steer_state_ = this->create_publisher<crs_msgs::msg::CarSteerState>("car_steer_state", 10);
}

bool WiFiCom::startUDPServer()
{
  udp_server_.setPortNum(udp_port_);
  if (!udp_server_.startListening())
  {
    RCLCPP_ERROR(this->get_logger(), "Could not start listening on UDP server!");
    return false;
  }
  RCLCPP_INFO(this->get_logger(), "WiFiCom: Started listening on UDPServer, port = %d", udp_port_);
  return true;
}

void WiFiCom::sendControlInput(const crs_msgs::msg::CarInput::SharedPtr msg)
{
  Packet p;
  SingleControlInput* inp = p.mutable_control_input();
  inp->set_torque_ref(msg->torque);
  inp->set_steer_ref(msg->steer);

  // Handle case where the car's internal steering map should be overriden and
  // a raw potentiometer reference sent. This is indicated by the steer_override
  // flag in the car_input message.
  if (msg->steer_override)
  {
    inp->mutable_steer_input()->set_steer_voltage(msg->steer);
  }
  else
  {
    inp->mutable_steer_input()->set_steer_angle(msg->steer);
  }

  std::string serialized_bytes;
  p.SerializeToString(&serialized_bytes);

  // Packets that are too long may be transferred in more than one transaction,
  // which is untested both in the server as in the client code.
  if (serialized_bytes.length() > UDP_MAX_RECOMMENDED_SIZE)
  {
    RCLCPP_WARN(this->get_logger(), "Packet length is %zuB, while recommended limit is %dB!",
                                     serialized_bytes.length(), UDP_MAX_RECOMMENDED_SIZE);
  }

  if (client_.ss_family == AF_INET || client_.ss_family == AF_INET6)
  {
    if (!udp_server_.send(&client_, (uint8_t*)serialized_bytes.c_str(), serialized_bytes.length()))
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to send packet!");
    }
  }
}

void WiFiCom::publishImuData(const IMUMeasurement& data)
{
  auto msg = sensor_msgs::msg::Imu();

  // Set header timestamp and frame_id
  msg.header.stamp = this->now();
  // TODO: Consider making the frame_id a configurable parameter
  msg.header.frame_id = "imu_link";

  // Convention: Set covariance of sensor measurement "orientation" to -1 if
  // this message field is invalid (which it is here, we don't have an absolute
  // orientation measurement yet). See ROS documentation for sensor_msgs/imu.
  msg.orientation_covariance[0] = -1;

  // Remap axes and adjust units:
  // - x and y axes need to be reversed
  // - acceleration must be converted from g to m/s^2
  // - angular velocity should be converted from deg/s to rad/s
  msg.linear_acceleration.x = -data.linear_acceleration().x() * GRAVITATIONAL_ACCELERATION;
  msg.linear_acceleration.y = -data.linear_acceleration().y() * GRAVITATIONAL_ACCELERATION;
  msg.linear_acceleration.z = +data.linear_acceleration().z() * GRAVITATIONAL_ACCELERATION;
  msg.angular_velocity.x = -data.angular_velocity().x() / 180 * M_PI;
  msg.angular_velocity.y = -data.angular_velocity().y() / 180 * M_PI;
  msg.angular_velocity.z = +data.angular_velocity().z() / 180 * M_PI;

  pub_imu_->publish(msg);
}

void WiFiCom::publishLowLevelControlInput(const SingleControlInput& reference, const MotorInput& drive_input,
                                          const MotorInput& steer_input)
{
  auto msg = crs_msgs::msg::CarLlControlInput();

  msg.drive_power = drive_input.power();
  msg.steer_power = steer_input.power();

  msg.steer_ref = reference.steer_ref();
  msg.torque_ref = reference.torque_ref();

  pub_car_ll_control_input_->publish(msg);
}

void WiFiCom::publishSteerState(const SteeringPositionMeasurement& steer_state)
{
  auto msg = crs_msgs::msg::CarSteerState();
  msg.steer_angle = steer_state.steer_rad();
  msg.steer_discrete_pos = steer_state.adc_meas();

  // This is not optimal yet, so it is not yet published.
  // Lower two bytes are the minimum position, upper two bytes are the maximum
  // position
  // msg.steer_min_pos = steer_state.adc_limits;
  // msg.steer_max_pos = steer_state.adc_limits;
  pub_car_steer_state_->publish(msg);
}