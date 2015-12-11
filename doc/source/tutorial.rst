Tutorial
=========

We can always discretize a hybridization function using the function *discretization.quick_map*,

Construct hybridization function
-------------------

1. It should be a function(any callable instance) with only one input variable :math:`\omega` and ouput variable :math:`{\cal D(\omega)}` which could be either a number of a square matrix.

2. The :math:`\omega` space is also important.
        .. hint::
            To create a simple plot in python::

                >>> from matplotlib import pyplot as plt
                >>> x_values = [1, 2, 3, 4]
                >>> y_values = [4, 2, 7, 3]
                >>> plt.plot(x_values, y_values)
                >>> plt.show()

    (c) From the above, estimate the central charge :math:`c` of the "Bethe phase" (1D quasi-long-range Néel phase) of the 1D Heisenberg model, and in light of that, think again about your answer to the last part of exercise 2.

        The formula for fitting the central charge on a system with open boundary conditions is:

        .. math::

            S = \frac{c}{6} \ln \left[ \frac{L}{\pi} \sin \left( \frac{\pi x}{L} \right) \right] + A

Using the right discretization mesh ticks
--------------------
