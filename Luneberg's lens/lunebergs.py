import numpy as np
# luneberg's lens 
import matplotlib.pyplot as plt

n0_luneberg = 1.0  # refractive index outside the lens
a = 1.0            # radius of luneberg lens

def deriv_fn_luneberg(state, n0, a):
    x, y, px, py = state
    r = np.sqrt(x**2 + y**2) + 1e-6

    if r <= a:
        n_luneberg = np.sqrt(2 - (r/a)**2)
        dn_dr = -r / (a**2 * n_luneberg)
    else:
        # refractive index is uniform outside the lens
        n_luneberg = n0_luneberg
        dn_dr = 0.0

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

    dx_ds = px / n_luneberg
    dy_ds = py / n_luneberg 
    dpx_ds = dn_dx
    dpy_ds = dn_dy

    return np.array([dx_ds, dy_ds, dpx_ds, dpy_ds])

def velocity_verlet_integrator(state):
    dt = 0.0001       # step size
    num_steps = 50000 # number of integration steps
    trajectory = []
    trajectory.append(np.array(state))

    for i in range(num_steps):
      deriv1 = deriv_fn_luneberg(state, n0_luneberg, a)
      x, y, px, py = state
      px_half = px + 0.5 * dt * deriv1[2]
      py_half = py + 0.5 * dt * deriv1[3] 

  
      state_half = np.array([x, y, px_half, py_half])
      deriv_half = deriv_fn_luneberg(state_half, n0_luneberg, a)

      x_new = x + dt * deriv_half[0]
      y_new = y + dt * deriv_half[1]

      state_new = np.array([x_new, y_new, px_half, py_half])
      deriv2 = deriv_fn_luneberg(state_new, n0_luneberg, a)
      px_new = px_half + 0.5 * dt * deriv2[2]
      py_new = py_half + 0.5 * dt * deriv2[3]

      state = np.array([x_new, y_new, px_new, py_new])
      trajectory.append(state)

      # stop if ray leaves simulation window
      if abs(x_new) > 3.0 or abs(y_new) > 3.0: 
        break

    return trajectory

ys = np.linspace(-0.8, 0.8, 10)
trajectories_luneberg = []

for y in ys:
  initial_state_luneberg = np.array([-3.0, y, 1.0, 0.0])
  trajectory_luneberg = np.array(velocity_verlet_integrator(initial_state_luneberg))
  trajectories_luneberg.append(trajectory_luneberg)

plt.figure(figsize=(12,12))

focus_y = []

for trajectory in trajectories_luneberg:
  x = trajectory[:, 0]
  y = trajectory[:, 1]

  # find point closest to lens boundary (x = a)
  idx = np.argmin(np.abs(x-a))
  focus_y.append(y[idx])

  plt.plot(x, y)

# computing variance - lower variance = better focusing
variance = np.var(focus_y)

print("Mean focus y-position =", np.mean(focus_y))
print("Standard deviation of focus y =", np.sqrt(variance))
print("Variance = ", variance)

focus_error = np.abs(np.array(focus_y) - np.mean(focus_y))
print("Max focus deviation =", np.max(focus_error))

# drawing luneberg's lens boundary
theta = np.linspace(0, 2 * np.pi, 500)
x_circle = a * np.cos(theta)
y_circle = a * np.sin(theta)
plt.plot(x_circle, y_circle, color = 'k' , linestyle='--', label = 'Luneberg\'s Lens', linewidth = 2)

# verifying the focus property
xp = np.linspace(-3, 3, 1000)
yp = np.zeros_like(xp)
plt.plot(xp, yp, 'k')

plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Luneberg Lens Ray Tracing')
plt.grid(True)
plt.legend()
plt.show()
