import os
import glob
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt


class GenericVehicle:
    """ A class for vehicles in the intercept scenario """

    def __init__(self, position=np.zeros((2, 1)), velocity=np.zeros((2, 1))):
        """ Constructor

        Args:
            position: The position vector
            velocity: The velocity vector
        """
        self.position = position
        self.velocity = velocity

    def update(self, timestep, acceleration=np.zeros((2, 1))):
        """ Update the vehicle state based on the current acceleration

        Args:
            timestep: The time step
            acceleration: The acceleration vector
        """

        self.position += timestep * self.velocity
        self.velocity += timestep * acceleration


class Vehicle(GenericVehicle):
    """ A class for the vehicle that is pursuing the target

    This vehicle will always start at the origin and use proportional
    navigation. These equations for the control law are taken from:

    Zarchan, Paul. (2012). Tactical and Strategic Missile Guidance (6th Edition).
    American Institute of Aeronautics and Astronautics. Retrieved from
    https://app.knovel.com/hotlink/toc/id:kpTSMGE001/tactical-strategic-missile/tactical-strategic-missile

    """

    def __init__(self, target):
        """ Constructor

        This assumes a normalized vehicle velocity (equal to 1)

        Args:
            target: The target vehicle
        """
        # Get the unit vector to start pointing towards the target
        super().__init__()
        self.target = target
        target_heading = np.arctan2(target.velocity[1], -target.velocity[0])
        target_velocity_magnitude = np.linalg.norm(target.velocity)
        self.heading_error = np.radians(np.random.uniform(-1, 1) * 5.)
        leading_angle = np.arcsin(
            target_velocity_magnitude * np.sin(target_heading + self.get_line_of_sight_angle()))
        vehicle_heading = leading_angle + self.heading_error + self.get_line_of_sight_angle()
        self.velocity = np.array([np.cos(vehicle_heading), np.sin(vehicle_heading)]).reshape(-1, 1)

    def get_line_of_sight_angle(self):
        """ Get the line of sight angle between the vehicle and target

        Returns:
            Line of sight angle
        """
        r_to_target = self.target.position - self.position
        return np.arctan2(r_to_target[1], r_to_target[0])

    def get_line_of_sight_angle_rate(self):
        """ Get line of sight angular rate

        Returns:
            Line of sight angular rate
        """
        r_to_target = self.target.position - self.position
        v_to_target = self.target.velocity - self.velocity
        return (r_to_target[0] * v_to_target[1] - r_to_target[1] * v_to_target[0]) / np.sum(
            r_to_target ** 2)

    def get_closing_velocity(self):
        """ Get the closing velocity

        Returns:
            The closing velocity between the vehicle and target
        """
        r_to_target = self.target.position - self.position
        v_to_target = self.target.velocity - self.velocity
        return -(r_to_target[0] * v_to_target[0] + r_to_target[1] * v_to_target[1]) / np.linalg.norm(
            r_to_target)

    def get_acceleration(self):
        """ Get the commanded acceleration given the current simulation state

        Returns:
            The commanded acceleration vector
        """
        gain = 3.
        command = gain * self.get_closing_velocity() * self.get_line_of_sight_angle_rate()
        acceleration = command * np.array(
            [-np.sin(self.get_line_of_sight_angle()), np.cos(self.get_line_of_sight_angle())]).reshape(-1, 1)
        return acceleration

    def update(self, timestep, acceleration=None):
        """ Update the vehicle to the next time step

        By default this will apply the line-of-sight proportional
        navigation control law to get the commanded acceleration

        Args:
            timestep: The time step
            acceleration: The vehicle acceleration vector
        """
        if acceleration is None:
            acceleration = self.get_acceleration()
        super().update(timestep, acceleration)


class Target(GenericVehicle):
    """ A target vehicle """

    def __init__(self, distance, velocity_magnitude, target_noise=0):
        """ Constructor

        Args:
            distance: The distance between the vehicle and target
            velocity_magnitude: The target velocity magnitude
        """
        angle = np.radians(45 + np.random.uniform(-1, 1) * target_noise)
        heading = np.pi + np.radians(np.random.uniform(-1, 1) * target_noise)
        position = distance * np.array([np.cos(angle), np.sin(angle)]).reshape(-1, 1)
        velocity = velocity_magnitude * np.array([np.cos(heading), np.sin(heading)]).reshape(-1, 1)
        super().__init__(position=position, velocity=velocity)


class Simulation:
    """ A class to run the simulation """

    def __init__(self, distance=10, velocity_ratio=1, target_noise=0, movie_filename='intercept.mpeg', savefig=False):
        """ Constructor

        Args:
            distance: The distance between the two vehicles
            velocity_ratio: The pursuit vehicle velocity normalized by the target velocity
        """
        self.count = 0
        self.time = 0
        self.target = Target(distance, 1 / velocity_ratio, target_noise=target_noise)
        self.movie_filename = movie_filename
        self.vehicle = Vehicle(self.target)
        self.savefig = savefig
        if self.savefig:
            self.fig, self.ax = plt.subplots()
            for ff in glob.glob('./snapshots/*.png') + glob.glob('*.mpeg'):
                print(ff)
                os.remove(ff)

    def run(self):
        """ Run the simulation """
        while self.vehicle.get_closing_velocity() >= 0:
            timestep = max(1e-1, float(5e-2 * self.r_to_target / self.vehicle.get_closing_velocity()))
            self.target.update(timestep)
            self.vehicle.update(timestep)
            self.plot(self.savefig)
            self.print_r_to_target()
            self.time += timestep
            self.count += 1

        if os.path.exists(self.movie_filename):
            os.remove(self.movie_filename)
        call(['ffmpeg', '-f', 'image2', '-i', 'snapshots/intercept_%06d.png', self.movie_filename])
        call(['open', self.movie_filename])

    @property
    def r_to_target(self):
        """ Get the distance to the target from the vehicle

        Returns:
            Distance to the target from the vehicle
        """
        r_to_target = self.target.position - self.vehicle.position
        return np.linalg.norm(r_to_target)

    def print_r_to_target(self):
        """ Print the distance to the target """
        r_to_target = self.r_to_target
        print("{:.3f}, {:.3f}, {:.3f}".format(self.time, r_to_target ** 2, np.linalg.norm(self.vehicle.get_acceleration())))

    def plot(self, savefig=False):
        """ Plot the state of the simulation

        This plots the vehicle and target position over time throughout the course of the intercept simulation
        """
        self.ax.scatter(self.vehicle.position[0], self.vehicle.position[1], c='b', marker='o', label='Vehicle')
        self.ax.scatter(self.target.position[0], self.target.position[1], c='r', marker='x', label='Target')
        self.ax.set_title('Time {:.3f}'.format(self.time))
        if self.time == 0:
            self.ax.legend()

        if savefig:
            self.fig.savefig('snapshots/' + self.movie_filename.replace('.mpeg', '_{:06d}.png'.format(self.count)))
