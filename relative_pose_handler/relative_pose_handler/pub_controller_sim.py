import rclpy
import numpy as np
import math
from rclpy.node import Node
from crs_msgs.msg import CarInput
from custom_msg.msg import TruckTrailerState
from custom_msg.msg import Shutdown
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        # self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        # self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        # state_value = F.relu(state_value)
        # action_value = F.relu(self.action_value(action))
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        # state_action_value = T.add(state_value, action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def save_checkpoint_progress(self, success):
        print(f'... saving checkpoint progress for {success}% success ...')

        # 1. Define the directory for this progress checkpoint
        progress_chkpt_dir = os.path.join(self.checkpoint_dir, str(success))
        os.makedirs(progress_chkpt_dir, exist_ok=True)

        # 2. Define the file path inside the new directory
        progress_checkpoint_file = os.path.join(progress_chkpt_dir, self.name + '_ddpg')

        # 3. Save the model to the new, correct path
        T.save(self.state_dict(), progress_checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best')
        T.save(self.state_dict(), checkpoint_file)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        # self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def save_checkpoint_progress(self, success):
        print(f'... saving checkpoint progress for {success}% success ...')

        # 1. Define the directory for this progress checkpoint
        progress_chkpt_dir = os.path.join(self.checkpoint_dir, str(success))
        os.makedirs(progress_chkpt_dir, exist_ok=True)

        # 2. Define the file path inside the new directory
        progress_checkpoint_file = os.path.join(progress_chkpt_dir, self.name + '_ddpg')

        # 3. Save the model to the new, correct path
        T.save(self.state_dict(), progress_checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best')
        T.save(self.state_dict(), checkpoint_file)


class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation , evaluate=False):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)

        if not evaluate:
            mu_prime = mu + T.tensor(self.noise(),
                                    dtype=T.float).to(self.actor.device)
        else:
            mu_prime = mu

        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def save_models_progress(self,success):
        self.actor.save_checkpoint_progress(success=success)
        self.target_actor.save_checkpoint_progress(success=success)
        self.critic.save_checkpoint_progress(success=success)
        self.target_critic.save_checkpoint_progress(success=success)

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)

class Controller_simulator(Node):
    def __init__(self):
        super().__init__('controller_publisher')
        self.publisher1 = self.create_publisher(CarInput,'control_input',10)
        self.shutdown_publisher = self.create_publisher(Shutdown, '/shutdown_signal', 10)
        self.sub1 = self.create_subscription(TruckTrailerState,'/truck_trailer_pose',self.controller_callback,10)
        self.timer = self.create_timer(0.001, self.timer_callback)
        self.agent = Agent(alpha=0.0001, beta=0.001,input_dims=[23], tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=1)
        self.state = [0, 0, 0, 0, 0, 0]
        self.max_map_x = 75
        self.min_map_y = -75
        self.min_map_x = -75
        self.max_map_y = 75
        self.workspace_width = self.max_map_x - self.min_map_x
        self.workspace_height = self.max_map_y - self.min_map_y
        self.max_expected_distance = np.sqrt(self.workspace_width ** 2 + self.workspace_height ** 2)
        self.goalx = 0
        self.goaly = -30
        self.goalyaw = np.deg2rad(90)
        self.steer = np.deg2rad(0)
        self.intit = False
        self.goal_reached = False
        self.steering = 0.0
        self.agent.load_models()

    def timer_callback(self):

        if self.goal_reached:
            return
        msg4 = CarInput()
        msg4.torque = -0.07 #  0.25 m/s
        msg4.velocity = math.nan
        msg4.steer =  self.steering # offfset due to physical constrain
        msg4.steer_override = False

        self.publisher1.publish(msg4)
        self.get_logger().info(str(msg4))

    def controller_callback(self, msg):

        if not self.goal_reached and msg.y2 < self.goaly :
            self.get_logger().info(f'Goal reached: trailer y ({msg.y2}) < goal y ({self.goaly}). Stopping publishing.')
            self.goal_reached = True
            shutdown_msg = Shutdown()
            self.shutdown_publisher.publish(shutdown_msg)
            self.get_logger().info('Published shutdown signal.')
            # Shutdown this node
            self.destroy_node()
            rclpy.shutdown()



        self.state[0] = msg.psi1
        self.state[1] = msg.psi2
        self.state[2] = msg.x1
        self.state[3] = msg.y1
        self.state[4] = msg.x2
        self.state[5] = msg.y2

        if not self.intit:

            observation = self.compute_observation(self.state,self.steer)
            self.intit = True
            action = self.agent.choose_action(observation, evaluate=True)
            scaled_action = np.clip(action,-1,1) * np.deg2rad(15)
            self.steering = scaled_action.item()
        else:
            observation = self.compute_observation(self.state, self.steering)
            action = self.agent.choose_action(observation, evaluate=True)
            scaled_action = np.clip(action,-1,1) * np.deg2rad(15)
            self.steering = scaled_action.item()


    def compute_observation(self, state, steering_angle):
        """Compute the enhanced observation vector from the current state."""
        # Extract state components
        psi_1, psi_2, x1, y1, x2, y2 = state

        # Normalize positions to [-1, 1] based on workspace center
        truck_x_norm = (x1 - (self.max_map_x + self.min_map_x) / 2) / (self.workspace_width / 2)
        truck_y_norm = (y1 - (self.max_map_y + self.min_map_y) / 2) / (self.workspace_height / 2)
        trailer_x_norm = (x2 - (self.max_map_x + self.min_map_x) / 2) / (self.workspace_width / 2)
        trailer_y_norm = (y2 - (self.max_map_y + self.min_map_y) / 2) / (self.workspace_height / 2)
        goal_x_norm = (self.goalx - (self.max_map_x + self.min_map_x) / 2) / (self.workspace_width / 2)
        goal_y_norm = (self.goaly - (self.max_map_y + self.min_map_y) / 2) / (self.workspace_height / 2)

        # Compute hitch-angle
        hitch_angle = psi_1 - psi_2

        # Compute distance to goal
        distance_to_goal = np.sqrt((x2 - self.goalx) * 2 + (y2 - self.goaly) * 2)
        distance_to_goal_norm = np.clip(distance_to_goal / self.max_expected_distance, 0, 1)

        # Compute angle from trailer to goal
        angle_to_goal = np.arctan2(self.goaly - y2, self.goalx - x2)

        # Compute orientation error
        orientation_error = self.goalyaw - psi_2

        # Compute position error in trailer's local coordinate frame
        dx_global = self.goalx - x2
        dy_global = self.goaly - y2
        dx_local = dx_global * np.cos(psi_2) + dy_global * np.sin(psi_2)
        dy_local = -dx_global * np.sin(psi_2) + dy_global * np.cos(psi_2)

        # Compute heading error
        heading_error = angle_to_goal - (psi_2 + np.deg2rad(180))

        # Normalize local errors
        dx_local_norm = np.clip(dx_local / self.max_expected_distance, -1, 1)
        dy_local_norm = np.clip(dy_local / self.max_expected_distance, -1, 1)

        # Construct observation vector
        observation = np.array([
            # Truck state (normalized position + angle as sin/cos)
            truck_x_norm,  # 0
            truck_y_norm,  # 1
            np.sin(psi_1),  # 2
            np.cos(psi_1),  # 3

            # Trailer state (normalized position + angle as sin/cos)
            trailer_x_norm,  # 4
            trailer_y_norm,  # 5
            np.sin(psi_2),  # 6
            np.cos(psi_2),  # 7

            # system dynamics (hitch angle as sin/cos) and (steering angle as sin/cos)
            np.sin(hitch_angle),  # 8
            np.cos(hitch_angle),  # 9
            np.sin(steering_angle),  # 10
            np.cos(steering_angle),  # 11

            # Goal (normalized position + angle as sin/cos)
            goal_x_norm,  # 12
            goal_y_norm,  # 13
            np.sin(self.goalyaw),  # 14
            np.cos(self.goalyaw),  # 15

            # Relative measurements
            distance_to_goal_norm,  # 16
            dx_local_norm,  # 17
            dy_local_norm,  # 18
            np.sin(orientation_error),  # 19
            np.cos(orientation_error),  # 20
            np.sin(heading_error),  # 21
            np.cos(heading_error)  # 22
        ], dtype=np.float32)

        return observation


def main():
    rclpy.init()
    controller_simulator = Controller_simulator()
    rclpy.spin(controller_simulator)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
