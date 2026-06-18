import numpy as np
import matplotlib.pyplot as plt

n0 = 1.5       # base index
alpha = 0.001   # gradient strength
dt = 1      # step size
num_steps = 1000 # number of integration steps

def deriv_fn(state, n0, alpha):

  x, y, px, py = state
  n = n0 + alpha * y
  
  dx_ds = px / n
  dy_ds = py / n
  dpx_ds = 0
  dpy_ds = alpha

  return np.array([dx_ds, dy_ds, dpx_ds, dpy_ds])

initial_state = [0, 0, 0.1, 0.1]

def euler_integrator(state):
  # state_{n+1} = state_n + dt * f(state_n)
  trajectory = []
  
  # appending starting point
  trajectory.append(np.array(state))

  for _ in range(num_steps):
    state = state + dt * deriv_fn(state, n0, alpha)
    trajectory.append(np.array(state))
  
  return trajectory

trajectory1 = np.array(euler_integrator(initial_state))

# plotting the trajectory using euler integration
x1 = trajectory1[:, 0]
y1 = trajectory1[:, 1]
plt.plot(x1, y1, linewidth = 2)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('Trajectory using Euler Integrator')
plt.show()

def rk4_integrator(state):
  trajectory = []

  # appending starting point
  trajectory.append(np.array(state))

  for _ in range(num_steps):

    k1 = deriv_fn(state, n0, alpha)
    k2 = deriv_fn(state + 0.5 * dt * k1, n0, alpha)
    k3 = deriv_fn(state + 0.5 * dt * k2, n0, alpha)
    k4 = deriv_fn(state + dt * k3, n0, alpha)

    state = state + dt * (k1 + 2*k2 + 2*k3 + k4)/6

    trajectory.append(np.array(state))

  return trajectory

trajectory2 = np.array(rk4_integrator(initial_state))

# plotting the trajectory using RK4 integration
x2 = trajectory2[:, 0]
y2 = trajectory2[:, 1]
plt.plot(x2, y2, linewidth = 2)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('Trajectory using RK4 Integrator')
plt.show()

def velocity_verlet_integrator(state):
    trajectory = []

    # appending starting point
    trajectory.append(np.array(state))

    for _ in range(num_steps):
      deriv1 = deriv_fn(state, n0, alpha)
      x, y, px, py = state

      # p_{n+1/2} = p_n + (dt/2) * F(x_n)
      px_half = px + 0.5 * dt * deriv1[2]
      py_half = py + 0.5 * dt * deriv1[3] 

      state_half = np.array([x, y, px_half, py_half])
      deriv_half = deriv_fn(state_half, n0, alpha)

      # x_{n+1} = x_n + dt * v(p_{n+1/2})
      x_new = x + dt * deriv_half[0]
      y_new = y + dt * deriv_half[1]

      state_new = np.array([x_new, y_new, px_half, py_half])
      deriv2 = deriv_fn(state_new, n0, alpha)

      # p_{n+1} = p_{n+1/2} + (dt/2) * F({x_{n+1}})
      px_new = px_half + 0.5 * dt * deriv2[2]
      py_new = py_half + 0.5 * dt * deriv2[3]

      state = np.array([x_new, y_new, px_new, py_new])
      trajectory.append(state)

    return trajectory

trajectory3 = np.array(velocity_verlet_integrator(initial_state))

# plotting the trajectory using velocity verlet integration 
x3 = trajectory3[:, 0]
y3 = trajectory3[:, 1]
plt.plot(x3, y3, linewidth = 2)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('Trajectory using Velocity Verlet Integrator')
plt.show()
