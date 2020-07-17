# Raytrace simulation of ocean acoustic propagation without boundary conditions
#
# By: Patrick Ledzian
# Date: 14Jul2020

# https://www.myphysicslab.com/explain/runge-kutta-en.html
# https://www.geeksforgeeks.org/runge-kutta-4th-order-method-solve-differential-equation/

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def raytrace(dist, state):
    # Mathematical representations of sound as a ray travelling through water. Approximates change in sound speed wrt
    # depth using the Munk sound speed profile. Assumes no boundary conditions.
    #
    # Inputs:
    # dist    -- scalar; distance / range in meters from the signal source; not used here, but ODE solver best practice
    # state   -- 2x1 array; first dimension is depth, second dimention is the change in depth wrt to range / distance
    #
    # Outputs:
    # return  -- 2x1 array; first order derivative of the state variable
    #

    # define c(z) using Munk sound speed profile
    z_prime = (2/1300) * (state[0] - 1300)
    eps = 0.00737
    c_z = 1500 * (1 + eps * (z_prime - 1 + np.exp(-z_prime)))

    # definition of c_z (the derivatice of c(z)
    dc_z = 1500 * eps * (2/1300) * (1 - np.exp(-z_prime))

    # update the state
    return np.array([state[1], ((1 + state[1]**2) * (-dc_z / c_z))])


def rungekutta(x0, y0, x, h):
    # ODE solver, good for a wide variety of physical phenomenon
    #
    # Inputs:
    # x0  --  scalar; initial position; distance / range in meters
    # y0  --  2x1 array; initial state; source depth in meters and initial launch angle of the signal in degrees
    # x   --  scalar; input range to solve for output; output is a depth
    # h   --  scalar; static step size of the simulation; # TODO: implement a variable step size approach
    #
    # Outputs:
    # y:  --  2x1 array; state output; first dimension is signal depth second dimension is rate of change of depth
    #

    n = int((x - x0) / h)
    y = y0
    for k in range(1, n + 1):
        k1 = h * raytrace(x0, y)                        # iteratively calls the differential equation to be solved
        k2 = h * raytrace(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * raytrace(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * raytrace(x0 + h, y + k3)

        # update the next value of y
        y = y + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # update the next value of x
        x0 = x0 + h

    return y


#
# Manual implementation of the runge-kutta ODE solver; note how much slower this is than the wrapper fcn
#

# simulation parameters
z_source = 1000                             # source depth in meters
theta_0 = np.linspace(-10, 10, 20)          # tangent of the launch angle in degrees
iters = np.shape(theta_0)[0]                # number of simulation variations to run
init_state = np.zeros((2, iters))                 # define empty array to be filled with initial conditions (computationally faster)
init_range = 0.0
step_size = 100.0

for i in range(0, iters):                   # build array of initial conditions
    theta = theta_0[i]
    init_state[0:2, i] = np.array([z_source, np.tan(theta * (np.pi/180))])          # initial state


r = 60000                                   # distance (range) in meters to simulate over
soln = np.zeros((iters+1, int(r/100)))      # array of solutions
for i in range(0, r, int(step_size)):
    if (i % 1000) == 0:
        print(i)
    soln[0, int(i/100)] = i/1000.0

    for j in range(0, iters):
        test_item = rungekutta(init_range, init_state[0:2, j], i, step_size)
        soln[j+1, int(i/100)] = test_item[0]

# #
# # Using the solve_ivp wrapper with RK45 as the ODE solver
# #
# range_span = [0, 50000]                    # simulation range / distance
# theta_0 = -18
# init_state = np.array([z_source, np.tan(theta_0 * (np.pi/180))])          # initial state
#
# # Run the naive simulation
# soln2 = solve_ivp(raytrace, range_span, init_state, method='RK45', dense_output=True)      # dense_output populates the "sol" object
# # print(soln.message)
# # print(soln)

# Plot the simulation results
fig = plt.figure()
ax = plt.axes()
ax.plot(soln[0, :], np.transpose(soln[1:, :]), linewidth=0.5)
ax.set_xlabel("Range (km)")
ax.set_ylabel("Depth (m)")
ax.set_title("Wave Propagation wrt Range vs Depth")
plt.gca().invert_yaxis()
plt.show()



# solvers
# [r,Z] = ODE45(@raytrace, [0 r_final], [z_source, tan(theta_0*pi/180)
# save_state = integrate.RK45(lambda range, state: raytrace(range, state), r, init, 1000)
# save_state = integrate.solve_ivp(lambda range, state: raytrace(range, state), (0, 1000), init, method="RK45", dense_output=True)