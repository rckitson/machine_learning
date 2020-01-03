#!/opt/anaconda3/bin/python3
""" Class definitions and a main routine to simulate traffic """
import os
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt


class Car:
    """ A class for modeling cars in traffic """

    def __init__(self, aggression=1.0, ahead=None):
        """ Constructor

        Args:
            aggression: The driver's aggressiveness
            ahead: The car ahead
        """

        self.aggression = aggression
        self.ahead = ahead
        self.position = 0
        self.velocity = 0

    def drive(self, time, time_step=0.001):
        """ Drive the car

        Args:
            time_step: The time step, seconds
        """

        displacement = (self.ahead.position - self.position)
        velocity_diff = (self.ahead.velocity - self.velocity)
        # Model driving as a linear spring and a linear damper
        force = self.aggression * (displacement + 0.25 * velocity_diff)
        self.position += time_step * self.velocity
        self.velocity += time_step * force


class Traffic:
    """ A class to simulate traffic """

    def __init__(self, length=1, average_speed=1):
        """ Constructor

        Args:
            length: The length of the traffic
            average_speed: The average speed of the traffic
        """

        self.length = length
        self.average_speed = average_speed

        self.time = 0
        self.cars = []
        self.positions = -1 * np.arange(self.length) / self.length * np.pi
        for ii in range(self.length):
            if ii > 0:
                self.cars.append(Car(aggression=1e1, ahead=self.cars[ii - 1]))
            elif ii == 0:
                self.cars.append(Car())

    def run(self, run_time, time_step=1e-3):
        """ Run the simulation 
        
        Args:
            run_time: The total number of time steps
            time_step: The time step
        """

        for _ in range(run_time):
            print("Time {} / {}".format(self.time, run_time))
            self.time += 1
            self.positions += time_step * self.average_speed
            for ii, car in enumerate(self.cars):
                if ii == 0:
                    car.position = -abs(self.positions[1]) / 4. * np.sin(0.5 * 2 * np.pi * self.time / run_time)
                else:
                    car.drive(time=self.time * time_step, time_step=time_step)
            self.plot()

    def plot(self):
        """ Plot the cars on a circle """
        R = self.length

        plt.figure()
        for ii, car in enumerate(self.cars):
            theta = self.positions[ii] + car.position
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            if ii == 0:
                plt.scatter(x, y, marker='x')
            else:
                plt.scatter(x, y)

        plt.axis('scaled')
        lim = (-1.2 * R, 1.2 * R)
        plt.ylim(lim)
        plt.xlim(lim)
        plt.savefig('traffic_{:d}.png'.format(self.time))
        plt.close()


if __name__ == "__main__":
    for ff in glob.glob('*.png'):
        os.remove(ff)
    Traffic(length=10, average_speed=2).run(100, time_step=5e-3)

    ffmpeg = '/usr/local/bin/ffmpeg'
    movie_filename = 'traffic.mp4'
    cmd = '-f image2 -i traffic_%d.png {}'.format(movie_filename).split(' ')
    if os.path.exists(movie_filename):
        os.remove(movie_filename)
    subprocess.call([ffmpeg] + cmd)
    subprocess.call(['open', movie_filename])
