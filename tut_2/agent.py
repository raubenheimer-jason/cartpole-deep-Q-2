
from itertools import count
import random
import torch
import gym
from dqn import DQN, ReplayMemory, Trainer, Transition
from plotting import plot_durations
from game import get_screen

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)


if __name__ == "__main__":
    env = gym.make('CartPole-v1').unwrapped

    env.reset()

    TARGET_UPDATE = 10

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # "online_net"
    policy_net = DQN(screen_height, screen_width, n_actions, device).to(device)
    target_net = DQN(screen_height, screen_width, n_actions, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # replay buffer
    memory = ReplayMemory()

    # from other youtube tut: need to initialise replay buffer with random movements before starting
    # 13:57 in tut p.1 (https://youtu.be/NP8pXZdU-5U?t=837)

    trainer = Trainer(device, memory, n_actions, policy_net, target_net)

    episode_durations = []

    # reward_buffer

    num_episodes = 30
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        # print(last_screen)
        # print(last_screen.shape)
        # break
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = trainer.select_action(state)
            # print(env.step(action.item()))
            _, reward, done, _, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            trainer.optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')

    while True:
        pass