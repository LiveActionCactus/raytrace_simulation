# Raytrace simulation of ocean acoustic propagation with boundary conditions
#
# By: Patrick Ledzian
# Date: 16Jul2020

# https://stackoverflow.com/questions/58261779/solving-ode-set-with-a-step-function-parameter-using-odeint
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
# https://stackoverflow.com/questions/56065467/finding-zero-crossing-in-python
# https://github.com/scipy/scipy/issues/9228
# https://stackoverflow.com/questions/60136367/simultaneous-event-detection-using-solve-ivp
# file:///tmp/mozilla_odysseus0/scipy-ode.pdf

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

def event_surface(dist, state):
    return (0 - state[0])


def event_bottom(dist, state):
    bottom = 5000               # bottom in meters
    return (state[0] - bottom)


# simulation parameters
z_source = 1000                             # source depth in meters
theta_0 = np.linspace(-18, 18, 20)
init_range_span = [0, 50000]                    # simulation range / distance
event_surface.terminal = True
event_surface.direction = 1                # very important that these values line up with the event function
event_bottom.terminal = True
event_bottom.direction = 1

# results = np.copy(theta_0)
# results = np.transpose(np.expand_dims(results, axis=(0, 1)))

results = list()
# run the simulation for all initial launch angles found in theta_0
for i in range(0, theta_0.shape[0]):
    init_state = np.array([z_source, np.tan(theta_0[i] * (np.pi/180))])          # initial state
    range_span = np.copy(init_range_span)
    r_out = np.array((init_range_span[0], init_range_span[0]))                # numpy won't concat 0-d arrays / scalars
    state_out = np.copy(init_state)
    while abs(r_out[-1] - range_span[1]) > 1.0:
        # run the ode solver
        soln = solve_ivp(raytrace, range_span, init_state, method='RK45', rtol=1e-13, atol=1e-14,
                         events=[event_surface, event_bottom], dense_output=True)      # dense_output populates the "sol" object

        N = len(soln.t)
        r_out = np.concatenate((r_out, soln.t[1:]))
        state_out = np.column_stack((state_out, soln.y[:, 1:]))

        range_span = [soln.t[N-1], range_span[1]]
        init_state = np.array((soln.y[0, N-1], soln.y[1, N-1]))

        # code gets stuck in an infinite loop hugging the boundaries without this; resolves numerical issue
        if (abs(event_surface(range_span[0], init_state)) < 1e-12) or (abs(event_bottom(range_span[0], init_state)) < 1e-12):
            init_state[1] = -init_state[1]

    r_out = r_out / 1000
    event = np.transpose(np.column_stack((r_out[1:], np.transpose(state_out))))
    results.append(event)
    # exit()

# TODO: generate smooth outputs using dense_output for each segment of the solution
# xeval = np.linspace(*range_span, 100)      # for creating a dense visual output

# Plot the simulation results
fig = plt.figure()
ax = plt.axes()
for i in range(0, theta_0.shape[0]):
    ax.plot(results[i][0][:], results[i][1][:], linewidth=0.5)
ax.set_xlabel("Range (km)")
ax.set_ylabel("Depth (m)")
ax.set_title("Wave Propagation wrt Range vs Depth")
plt.gca().invert_yaxis()
plt.show()