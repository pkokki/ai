"""https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/"""
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer


def naive_sum_reward_agent(env, num_episodes=500):
    """
    policy π which maps states to actions in an optimal way to maximize reward:
    choose the action resulting in the greatest previous cumulative reward
    :param env: the gym environment
    :param num_episodes: the number of episodes (or number of games) that we will train the agent
    :return: the 5x2 reward table
    :example: [[     0. 561368.]
               [ 28086.      0.]
               [     0.  90030.]
               [     0.  17776.]
               [ 96382.      0.]]
    """
    # the table that will hold our cumulative rewards for each action in each state
    reward_table = np.zeros((5, 2))
    for g in range(num_episodes):
        current_state = env.reset()
        done = False
        while not done:
            # if there are not any existing values in the reward_table for the current state
            if np.sum(reward_table[current_state, :]) == 0:
                # make a random selection of action
                selected_action = np.random.randint(0, 2)
            else:
                # select the action with highest cumulative reward
                selected_action = np.argmax(reward_table[current_state, :])
            # the selected action is fed into the environment
            new_state, reward, done, _ = env.step(selected_action)
            # update reward table cell by adding the reward to cell value
            reward_table[current_state, selected_action] += reward
            # update current state
            current_state = new_state
    return reward_table


def q_learning_with_table(env, num_episodes=500):
    """

    :param env:
    :param num_episodes:
    :return:
    """
    # Q(s,a) += learning_rate * (reward + discounting_factor * max[Q(s',a')] - Q(s,a))
    q_table = np.zeros((5, 2))
    discounting_factor = 0.95
    learning_rate = 0.8
    for i in range(num_episodes):
        current_state = env.reset()
        done = False
        while not done:
            if np.sum(q_table[current_state, :]) == 0:
                # make a random selection of actions
                action = np.random.randint(0, 2)
            else:
                # select the action with largest q value in state s
                action = np.argmax(q_table[current_state, :])
            new_state, reward, done, _ = env.step(action)
            q_table[current_state, action] += reward + learning_rate * \
                (discounting_factor * np.max(q_table[new_state, :]) - q_table[current_state, action])
            current_state = new_state
    return q_table


def eps_greedy_q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        while not done:
            # select the action with highest cumulative reward
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s, :])
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table


def test_methods(env, num_iterations=100, num_episodes=500):
    winner = np.zeros((3,))
    m0_table = naive_sum_reward_agent(env, num_episodes)
    m1_table = q_learning_with_table(env, num_episodes)
    m2_table = eps_greedy_q_learning_with_table(env, num_episodes)
    for g in range(num_iterations):
        m0 = run_game(m0_table, env)
        m1 = run_game(m1_table, env)
        m2 = run_game(m2_table, env)
        w = np.argmax(np.array([m0, m1, m2]))
        winner[w] += 1
        print("Game {} of {}: winner={}".format(g + 1, num_iterations, w))
    return winner


def run_game(table, env):
    s = env.reset()
    tot_reward = 0
    done = False
    while not done:
        a = np.argmax(table[s, :])
        s, r, done, _ = env.step(a)
        tot_reward += r
    return tot_reward


def q_learning_with_keras(env, num_episodes=500):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 5)))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
            new_s, r, done, _ = env.step(a)
            target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
            target_vec = model.predict(np.identity(5)[s:s + 1])[0]
            target_vec[a] = target
            model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / 1000)
    return model


episodes = 20
chain_env = gym.make('NChain-v0')
print('A first naive heuristic for reinforcement learning')
nr = naive_sum_reward_agent(chain_env, episodes)
print(nr)

print('Delayed reward reinforcement learning (Q learning)')
ql = q_learning_with_table(chain_env, episodes)
print(ql)

print('Q learning with ϵ-greedy action selection')
eql = eps_greedy_q_learning_with_table(chain_env, episodes)
print(eql)

# wins = test_methods(chain_env, 100, episodes)
# print(wins)

print('Reinforcement learning with Keras')
keras_model = q_learning_with_keras(chain_env, episodes)
