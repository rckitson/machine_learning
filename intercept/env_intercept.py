import pyglet
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np

import intercept


class InterceptEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, savefig=False, savefig_format='image_{:06d}.png'):
        # Limits on the commanded acceleration
        self.max_acceleration = 0.05

        # Plotting
        self.viewer = None
        self.bounds0 = None
        self.vehicle_transform = None
        self.acceleration_vector_transform = None
        self.target_transform = None
        self.u = None
        self.savefig = savefig
        self.savefig_count = 0
        self.savefig_format = savefig_format

        # Environment simulation, vehicle, and target from the intercept module
        self.sim = None

        # Global time step
        self.timestep = 0.1

        # The action is the commanded acceleration vector of the intercepting vehicle
        action_high = self.max_acceleration
        # The observation space includes the position and velocity of the vehicle and target in 2D space
        observation_high = np.array([10., 10.])
        self.action_space = spaces.Box(low=-action_high, high=action_high, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        """ Step the environment by one timestep

        Args:
            u: The commanded acceleration

        Returns:
            The state, reward, done flag, info
        """
        acceleration_direction = self.sim.vehicle.get_line_of_sight_angle() + np.pi / 2.
        self.u = np.clip(u, -self.max_acceleration, self.max_acceleration) * np.array(
            [np.cos(acceleration_direction), np.sin(acceleration_direction)])

        self.sim.target.update(self.timestep)
        self.sim.vehicle.update(self.timestep, self.u)

        done = bool(self.sim.vehicle.get_closing_velocity() <= 0)
        if done:
            reward = -(self.sim.r_to_target ** 2)
        else:
            reward = 0
        return self._get_obs(), reward, done, {}

    def reset(self):
        """ Reset the environment """
        self.close()
        self.sim = intercept.Simulation(100, 2, target_noise=5)
        return self._get_obs()

    def _get_obs(self):
        # Observe the relative position and velocity
        state = (self.sim.vehicle.get_closing_velocity(), self.sim.vehicle.get_line_of_sight_angle_rate())
        return np.array(state).ravel()

    def render(self, mode='human'):
        """ Render one frame of the simulation environment

        Args:
            mode: (defaults to human) render to the current display or terminal and
                return nothing
        """

        pos_min = np.minimum(self.sim.vehicle.position, self.sim.target.position).ravel()
        pos_max = np.maximum(self.sim.vehicle.position, self.sim.target.position).ravel()
        pos_range = np.max(pos_max - pos_min)

        bounds = (pos_min[0] - pos_range / 2., pos_max[0] + pos_range / 2.,
                  pos_min[1] - pos_range / 2., pos_max[1] + pos_range / 2.)
        radius = self.sim.r_to_target / 50.
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.bounds0 = bounds
            self.viewer.set_bounds(self.bounds0[0], self.bounds0[1], self.bounds0[2], self.bounds0[3])

            vehicle = rendering.make_circle(radius=radius)
            vehicle.set_color(0., 0., 1.)
            self.vehicle_transform = rendering.Transform()
            vehicle.add_attr(self.vehicle_transform)
            self.viewer.add_geom(vehicle)

            target = rendering.make_circle(radius=radius)
            target.set_color(1., 0., 0.)
            self.target_transform = rendering.Transform()
            target.add_attr(self.target_transform)
            self.viewer.add_geom(target)

        bounds_min = np.minimum(bounds, self.bounds0)
        bounds_max = np.maximum(bounds, self.bounds0)
        self.bounds0 = (bounds_min[0], bounds_max[1], bounds_min[2], bounds_max[3])
        self.viewer.set_bounds(self.bounds0[0], self.bounds0[1], self.bounds0[2], self.bounds0[3])

        acceleration_angle = np.arctan2(self.u[1], self.u[0])
        vector_length = 10 * radius * np.linalg.norm(self.u) / self.max_acceleration
        acceleration_vector = rendering.make_polyline(
            v=[[0, 0], [vector_length * np.cos(acceleration_angle), vector_length * np.sin(acceleration_angle)]])
        acceleration_vector.set_linewidth(10)
        self.acceleration_vector_transform = rendering.Transform()
        self.acceleration_vector_transform.set_translation(self.sim.vehicle.position[0], self.sim.vehicle.position[1])
        acceleration_vector.add_attr(self.acceleration_vector_transform)
        self.viewer.add_onetime(acceleration_vector)

        self.vehicle_transform.set_translation(self.sim.vehicle.position[0], self.sim.vehicle.position[1])
        self.target_transform.set_translation(self.sim.target.position[0], self.sim.target.position[1])

        rendered_viewer = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        if self.savefig:
            filename = self.savefig_format.format(self.savefig_count)
            pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
            self.savefig_count += 1
        return rendered_viewer

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
