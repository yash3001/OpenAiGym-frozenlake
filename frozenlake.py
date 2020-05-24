import numpy as np
import gym
import random
import time
from os import system

env = gym.make("FrozenLake-v0")

state_space_size = env.observation_space.n # x--> 16
action_space_size = env.action_space.n # y --> 4

q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1  # --- alpha
discount_rate = 0.99 # --- gamma

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# q learning algo
for episode in range(num_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        # exploration-exploitation trade off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] + learning_rate*(reward + discount_rate*np.max(q_table[new_state, :])-q_table[state, action])

        state = new_state
        rewards_current_episode += reward

        if done == True:
            break

    #exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print("***********Average reward per thousand episodes*****************\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n************Q-Table*********************\n")
print(q_table)

# Playing
for episode in range(5):
    state = env.reset()
    done = False
    print("**********EPISODE ", episode+1, "***************\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        _ = system('clear')
        env.render()
        time.sleep(0.1)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        if done:
            _ = system('clear')
            env.render()
            if reward == 1:
                print("*******You reached the goal!*********")
                time.sleep(2)

            else:
                print("********You fell!***********")
                time.sleep(2)
            _ = system('clear')
            break

        state = new_state

env.close()
