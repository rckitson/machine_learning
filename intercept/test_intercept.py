from intercept import Simulation
import numpy as np

# np.random.seed(100)
if __name__ == "__main__":
    sim = Simulation(40, 2, savefig=True)
    sim.run()
