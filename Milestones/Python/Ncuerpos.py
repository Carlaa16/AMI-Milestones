import matplotlib.pyplot as plt
import numpy as np
from numpy import array, zeros, linspace, reshape

Nb = 2 # Numero de cuerpos
Nc = 2 # Numero de coordenadas

def F(U, t): 
    pU = reshape(U, (Nb, Nc, 2)) # Nb= numero de cuerpos, Nc= numero de coordenadas/componentes , 2 = velocidad y posicion
    r = reshape(pU[:, :, 0], (Nb, Nc)) # vector posicion (0), apunta a todos los cuerpos y componentes 
    v = reshape(pU[:, :, 1], (Nb, Nc)) # vector velocidad (1), apunta a todos los cuerpos y coordenadas
    # r y v son punteros de punteros
    Fs = zeros(2*Nb*Nc) # Fuerza sobre todos los cuerpos y componentes
    pFs = reshape(Fs, (Nb, Nc, 2)) # tengo que usar la misma estructura que U
    drdt = reshape(pFs[:, :, 0], (Nb, Nc)) # vector aceleracion (0), apunta a todos los cuerpos y componentes
    dvdt = reshape(pFs[:, :, 1], (Nb, Nc)) # vector aceleracion (1), apunta a todos los cuerpos y componentes
    drdt = v # La derivada de la posicion es la velocidad
    for i in range(Nb):
        for j in range(Nb):
            if i != j:
                rij = r[j] - r[i]
                rij_norm = np.sqrt(np.sum(rij**2))
                rij3 = rij_norm**3
                dvdt[i] += rij/rij3 # Ley de gravitacion universal
    return Fs

# Definir las condiciones iniciales
# U tiene la forma [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
U0 = array([1, 0, 0, 1, -1, 0, 0, -1])  # Ejemplo de condiciones iniciales para 2 cuerpos
t = 0  # Tiempo inicial

# Calcular las fuerzas
fuerzas = F(U0, t)

# Imprimir las fuerzas
print("Fuerzas calculadas:")
print(fuerzas)

# Opcional: Graficar las fuerzas
pFs = reshape(fuerzas, (Nb, Nc, 2))
for i in range(Nb):
    plt.quiver(U0[2*i], U0[2*i+1], pFs[i, 0, 1], pFs[i, 1, 1], angles='xy', scale_units='xy', scale=1, color='r')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fuerzas sobre los cuerpos')
plt.grid()
plt.show()


def integrate(U0, t_max, dt):
    t_values = np.arange(0, t_max, dt)
    U_values = zeros((len(t_values), len(U0)))
    U_values[0] = U0
    for k in range(1, len(t_values)):
        U = U_values[k-1]
        dU = F(U, t_values[k-1])
        U_new = U + dt * dU
        U_values[k] = U_new
    return t_values, U_values


t_max = 10  # Tiempo máximo de simulación
dt = 0.01  # Paso de tiempo

# Integrar las ecuaciones de movimiento
t_values, U_values = integrate(U0, t_max, dt)

# Graficar las órbitas
plt.figure(figsize=(10, 10))
for i in range(Nb):
    x = U_values[:, 2*i]
    y = U_values[:, 2*i+1]
    plt.plot(x, y, label='Cuerpo ' + str(i+1))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Órbitas de los cuerpos')
plt.legend()
plt.grid()
plt.show()