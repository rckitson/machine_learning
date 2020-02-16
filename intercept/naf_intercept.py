import os
import glob
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import GaussianWhiteNoiseProcess
from rl.core import Processor

import env_intercept


class InterceptProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward


# Get the environment and extract the number of actions.
env = env_intercept.InterceptEnv()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.shape[0]

nodes = 32
layers = 4

# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
for _ in range(layers):
    V_model.add(Dense(nodes))
    V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
for _ in range(layers):
    mu_model.add(Dense(nodes))
    mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
for _ in range(layers):
    x = Dense(nodes)(x)
    x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = InterceptProcessor()
memory = SequentialMemory(limit=100000, window_length=1)
random_process = GaussianWhiteNoiseProcess(mu=0., sigma=1.0, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=10000, random_process=random_process,
                 gamma=0.99, target_model_update=1e-3, processor=processor)

save_filename = 'weights_naf_intercept.h5f'

print(os.path.exists(save_filename))
if not os.path.exists(save_filename):
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
else:
    agent.compile(Adam(lr=1e-4, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if os.path.exists(save_filename):
    print('loading weights')
    try:
        agent.load_weights(save_filename)
    except:
        print('Could not load the weights')
        pass
agent.fit(env, nb_steps=int(1e2), visualize=True, verbose=1, nb_max_episode_steps=None, action_repetition=10)
agent.save_weights(save_filename, overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
for ff in glob.glob('snapshots/env_intercept_*.png'):
    os.remove(ff)
env.savefig = True
env.savefig_format = 'snapshots/env_intercept_{:06d}.png'
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=2000)
