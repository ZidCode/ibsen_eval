from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

# Argumentwerte als 1D Arrays erzeugen
x_1d = np.linspace(-3,3,601)
y_1d = np.linspace(-2,2,401)

# Argumentwerte als 2D Arrays erzeugen
x_2d, y_2d = np.meshgrid(x_1d, y_1d)

# Interessante Daten erzeugen
z_2d = 1/(x_2d**2 + y_2d**2 + 1) * np.cos(np.pi * x_2d) * np.cos(np.pi * y_2d)


z_matrix = np.arange(5)
for i in range(4):
    z_matrix = np.vstack((z_matrix, np.arange(5) + (i + 1)))
x_vector = np.arange(5)
y_vector = np.arange(5)
f = interpolate.interp2d(x_vector, y_vector, z_matrix)

x_new = np.linspace(0,4,101)
y_new = np.linspace(0,4,101)
z_new = f(x_new, y_new)
plt.figure()
plt.pcolormesh(x_new, y_new, z_new)
plt.gca().set_aspect("equal")
plt.xlabel(r'Coverty factor $\Delta$ $g_{dsr}$')
plt.ylabel(r'Coverty factor $\Delta$ $g_{dsa} \left[ \%\right]$')
plt.colorbar(label=r'$\Delta \alpha \; \left[ \%\right]$')
plt.show()
