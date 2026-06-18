# maxwell's fish eye lens
import numpy as np 
import matplotlib.pyplot as plt 

n0_fisheye = 1.5 
a = 1.0

def deriv_fn_fisheye(state, n0, a):
  x, y, px, py = state
  r = np.sqrt(x**2 + y**2) + 1e-6     # to avoid division by zero
  n = n0 / (1 + (r / a)**2)

  dn_dr = -2 * n0 * r / (a**2 * (1 + (r / a)**2)**2)
  dn_dx = dn_dr * x / r
  dn_dy = dn_dr * y / r

  '''
  px = n(r) * vx 
  py = n(r) * vy
  dx/ds = px / n(r)
  dy/ds = py / n(r)
  dpx/ds = dn_dx
  dpy/ds = dn_dy

  '''

  dx_ds = px / n
  dy_ds = py / n
  dpx_ds = dn_dx
  dpy_ds = dn_dy

  return np.array([dx_ds, dy_ds, dpx_ds, dpy_ds])

def rk4_integrator_fisheye(state, L1):
  '''
  rk4 integration for ray propagation and tracking optical angular momentum
  using L = x * py - y * px = constant 
  '''

  dt = 0.0001
  num_steps = 35000

  trajectory = []
  trajectory.append(np.array(state))

  max_error = 0.0

  for i in range(num_steps):
    k1 = deriv_fn_fisheye(state, n0_fisheye, a)
    k2 = deriv_fn_fisheye(state + 0.5 * dt * k1, n0_fisheye, a)
    k3 = deriv_fn_fisheye(state + 0.5 * dt * k2, n0_fisheye, a)
    k4 = deriv_fn_fisheye(state + dt * k3, n0_fisheye, a)

    state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    x, y, px, py = state
    L = x * py - y * px 

    # storing the max error of all states
    max_error = max(max_error, abs(L-L1))

    trajectory.append(np.array(state))

  return np.array(trajectory), max_error

# initial conditions
angles = np.linspace(0, 2 * np.pi, 24, endpoint = False)
trajectories = []

# source position - off centre to avoid trivial symmetric case
x0, y0 = 0.5, 0.0
r0 = np.sqrt(x0**2 + y0**2) + 1e-6

# refractive index at source position
n0_init = n0_fisheye / (1 + (r0 / a)**2)

global_max_error = 0.0
global_max_relative_error = 0.0

for theta in angles:
  initial_state_fisheye = np.array([x0, 
                                    y0, 
                                    n0_init * np.cos(theta), 
                                    n0_init * np.sin(theta)])
  
  x, y, px, py = initial_state_fisheye
  L1 = x * py - y * px 
  
  trajectory_fisheye, max_error = rk4_integrator_fisheye(initial_state_fisheye, L1)

  # max error of any state of all simulations
  global_max_error = max(global_max_error, max_error)

  # to avoid relative error computation when angular momentum is ~0
  if abs(L1) > 1e-12:
        # to normalize the deviation in angular momentum conservation
        rel_error = max_error / abs(L1)
        global_max_relative_error = max(
            global_max_relative_error,
            rel_error
        )

  trajectories.append(trajectory_fisheye)

print("Global maximum error is: ", global_max_error)
print("Relative error is: ", global_max_relative_error)

plt.figure(figsize=(8,8))
plt.scatter([x0], [y0], s=130, c='red', label='Source')

# plotting the trajectory
for trajectory in trajectories:
  x = trajectory[:, 0]
  y = trajectory[:, 1]
  plt.plot(x, y)

plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Maxwell Fish-Eye Lens Ray Tracing")
plt.grid(True)
plt.legend()
plt.show()
