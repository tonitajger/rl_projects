from keras.models import Sequential
from keras.layers import Dense
import time
import math
import random

import numpy as np
import pickle
import gym


class DeepCartPole:
    """
    RL classic Q-learning implementation of cart pole problem, balancing a pole upright on a cart on a friction-less
    surface. The "Gym" package from OpenAI is used as the gym environment for the RL method. This simple Q-learning
    method is quite inefficient and under-performing, but illustrates the essential idea of the RL approach.
    """

    def __init__(self):
        """
        Initiate the CartPole object and the Q-matrix with 0s
        """
        self.Q = np.zeros((10000, 2))

    def train(self, n, save_file=None, monitor=False, printout=False):
        """
        Training the agent on the cart pole environment. The environment is reset at time 200, if the angle of the pole
        exceeds +-12 degrees or cart position outside +-2.4.
        :param n: Number of training episodes
        :param save_file: If the Q-table should be saved or not to .pkl file
        :param monitor: If the environment should be rendered or not
        :return: None
        """
        env = gym.make('CartPole-v0')

        # Initial hyper parameter values
        eps = .5
        alpha = 1
        gamma = .99

        # Save the times to a 1000 dim vector for calculating the average to monitor the training
        avg = np.zeros(1000)
        for i_episode in range(1, n):

            # Alpha and epsilon decreasing with the number of episodes
            if alpha > .2:
                alpha *= .9995
            if eps > .1:
                eps *= .9996

            # Get the initial state of the cart pole
            observation = env.reset()
            state = get_state(np.array(observation))

            for t in range(1, 1000):

                # Monitor every 10th episode if monitor param is True
                if monitor and (i_episode % 100 == 0):
                    env.render()
                    time.sleep(.02)

                # Random variable to decide if the agent should explore or exploit
                explore = random.random()
                if explore < eps:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state])

                    # If the maximum is 0 i.e. no value has been set yet to the row, choose action at random
                    if not self.Q[state, action]:
                        action = random.randint(0, 1)

                # Take the decided action and time step
                observation, reward, done, info = env.step(action)
                new_state = get_state(np.array(observation))

                # Tweaking of the reward given: -1 if the pole falls and larger reward when pole angle close to 0
                reward -= 5*abs(observation[2])

                # Update the Q-value
                self.Q[state, action] = self.Q[state, action] + alpha*(reward + gamma*np.amax(self.Q[new_state]) -
                                                                       self.Q[state, action])
                if done:
                    avg[i_episode % 1000] = t
                    if printout and (i_episode % 1000 == 0):
                        print("Iteration: {}".format(i_episode))
                        print("Episode finished at average time t = {}".format(np.mean(avg)))
                    break

                state = new_state

        # Save the Q-table
        if save_file:
            q_file = open(save_file, "wb")
            pickle.dump(self.Q, q_file)
            q_file.close()

        env.close()

    def demo(self, n):
        """
        Demonstrates the cart pole performance using the loaded Q-matrix
        :param n: Number of episodes to demonstrate
        :return: None
        """
        env = gym.make('CartPole-v0')

        for i in range(n):
            observation = env.reset()
            state = get_state(np.array(observation))

            for t in range(1, 1000):
                env.render()
                time.sleep(.02)

                # Choose action from Q-table
                action = np.argmax(self.Q[state])
                observation, reward, done, info = env.step(action)
                state = get_state(np.array(observation))

                if done:
                    print("Episode finished after {} time steps".format(t + 1))
                    break
        env.close()

    def load_q_file(self, file_name):
        """
        Load a Q-matrix using a saved .pkl file
        :param file_name: Path to the saved Q-matrix file
        :return: None
        """
        f = open(file_name, "rb")
        self.Q = pickle.load(f)
        f.close()


if __name__ == "__main__":
    cp = CartPole()
    # cp.train(10000, save_file="Q_tables/Q1e4states.pkl", monitor=False)
    cp.load_q_file("cartpole/Q_tables/Q1e4states.pkl")
    cp.demo(5)
