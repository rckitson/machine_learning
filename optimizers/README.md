# Optimizers

Some optimization schemes from the literature that I decided to 
test out and compare the convergence characteristics.

I implemented to test functions, the Rosenbrock function and
the circle function. The plot below is the result of minimizing circle function in 
one dimension, which is a parabola. 

The driver script test_optimizer.py can be modified to test out
higher dimensions of the test functions and different hyperparameters
such as the learning rate.

Overall, the methods were straightforward to implement and I learned a lot
of what is going on behind the scenes in the machine learning libraries.

Convergence history:
![convergence history](./convergence_history.png)
