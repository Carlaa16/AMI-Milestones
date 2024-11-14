import numpy as np
from scipy.optimize import newton

def Euler(U, dt, t, F):
    return U + dt * F(U, t)

def Euler_inverso(U, dt, t, F):
    def G(X):
        return X - U - dt * F(X, t)
    return newton(G, U)

def Crank_Nicolson(U, dt, t, F):
    def G(X):
        return X - U - dt/2 * (F(X, t) + F(U, t))
    return newton(G, U)

def Kepler(U, t):
    x = U[0]; y = U[1]; dxdt = U[2]; dydt = U[3]
    d = (x**2 + y**2)**1.5
    return np.array([dxdt, dydt, -x/d, -y/d])

def linear_oscillator(U, t):
    x = U[0]
    v = U[1]
    return np.array([v, -x])

def solve_oscillator(method, U0, dt, t_max, F):
    t_values = np.arange(0, t_max, dt)
    U_values = [U0]
    for t in t_values[:-1]:
        U_new = method(U_values[-1], dt, t, F)
        U_values.append(U_new)
    return t_values, np.array(U_values)

# Parámetros de la simulación
U0 = np.array([1, 0])  # Condiciones iniciales: x(0) = 1, v(0) = 0
dt = 0.01
t_max = 10

# Resolver usando diferentes métodos
t_values, U_euler = solve_oscillator(Euler, U0, dt, t_max, linear_oscillator)
t_values, U_euler_inverso = solve_oscillator(Euler_inverso, U0, dt, t_max, linear_oscillator)
t_values, U_crank_nicolson = solve_oscillator(Crank_Nicolson, U0, dt, t_max, linear_oscillator)

# Graficar los resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(t_values, U_euler[:, 0], label='Euler')
plt.plot(t_values, U_euler_inverso[:, 0], label='Euler Inverso')
plt.plot(t_values, U_crank_nicolson[:, 0], label='Crank-Nicolson')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.title('Linear Oscillator Solutions with Different Numerical Methods')
plt.show()


##############################################################################################################
##############################################################################################################
#######################         REGIONES DE ESTABILIDAD         ##############################################
##############################################################################################################
##############################################################################################################



def stability_region_euler():
    z = np.linspace(-3, 3, 400) + 1j * np.linspace(-3, 3, 400)[:, None]
    G = 1 + z
    return np.abs(G) <= 1

def stability_region_euler_inverso():
    z = np.linspace(-3, 3, 400) + 1j * np.linspace(-3, 3, 400)[:, None]
    G = 1 / (1 - z)
    return np.abs(G) <= 1

def stability_region_crank_nicolson():
    z = np.linspace(-3, 3, 400) + 1j * np.linspace(-3, 3, 400)[:, None]
    G = (1 + z / 2) / (1 - z / 2)
    return np.abs(G) <= 1

# Graficar las regiones de estabilidad
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(stability_region_euler(), extent=[-3, 3, -3, 3], origin='lower', cmap='Greys')
plt.title('Región de Estabilidad - Euler')
plt.xlabel('Re(hλ)')
plt.ylabel('Im(hλ)')

plt.subplot(1, 3, 2)
plt.imshow(stability_region_euler_inverso(), extent=[-3, 3, -3, 3], origin='lower', cmap='Greys')
plt.title('Región de Estabilidad - Euler Inverso')
plt.xlabel('Re(hλ)')
plt.ylabel('Im(hλ)')

plt.subplot(1, 3, 3)
plt.imshow(stability_region_crank_nicolson(), extent=[-3, 3, -3, 3], origin='lower', cmap='Greys')
plt.title('Región de Estabilidad - Crank-Nicolson')
plt.xlabel('Re(hλ)')
plt.ylabel('Im(hλ)')

plt.tight_layout()
plt.show()

##############################################################################################################
##############################################################################################################
##################        EXPLICACION RESULTADOS NUMERICOS         ###########################################
##############################################################################################################
##############################################################################################################

"""
Para explicar los resultados numéricos en base a las regiones de estabilidad absoluta, es 
importante entender cómo la estabilidad de un método numérico afecta la precisión y la
convergencia de la solución. Aquí hay algunos puntos clave para considerar:

Método de Euler:
    La región de estabilidad del método de Euler es limitada y solo incluye una pequeña 
    porción del plano complejo alrededor del origen. Esto significa que el método de Euler es
    condicionalmente estable y solo puede ser utilizado con pasos de tiempo pequeños para 
    mantener la estabilidad. En la práctica, si el paso de tiempo (h) es demasiado grande, la 
    solución numérica puede divergir rápidamente, lo que se observa como oscilaciones crecientes
    en la solución.
Método de Euler Inverso:
    La región de estabilidad del método de Euler inverso incluye todo el semiplano izquierdo del 
    plano complejo. Esto significa que el método de Euler inverso es incondicionalmente estable 
    para problemas lineales con valores propios negativos. En la práctica, esto permite utilizar
    pasos de tiempo más grandes sin que la solución numérica diverja, aunque puede introducir 
    errores de truncamiento si el paso de tiempo es demasiado grande.
Método de Crank-Nicolson:
    La región de estabilidad del método de Crank-Nicolson incluye todo el semiplano izquierdo 
    del plano complejo, similar al método de Euler inverso. Esto significa que el método de 
    Crank-Nicolson es incondicionalmente estable para problemas lineales con valores propios 
    negativos. En la práctica, el método de Crank-Nicolson es más preciso que el método de 
    Euler inverso debido a su naturaleza implícita y su tratamiento simétrico del tiempo, lo 
    que reduce los errores de fase.

Ejemplo de Análisis Numérico
Supongamos que resolvemos el oscilador lineal (\ddot{x} + x = 0) con diferentes métodos y 
comparamos los resultados:
"""

# Parámetros de la simulación
U0 = np.array([1, 0])  # Condiciones iniciales: x(0) = 1, v(0) = 0
dt = 0.1  # Paso de tiempo
t_max = 10

# Resolver usando diferentes métodos
t_values, U_euler = solve_oscillator(Euler, U0, dt, t_max, linear_oscillator)
t_values, U_euler_inverso = solve_oscillator(Euler_inverso, U0, dt, t_max, linear_oscillator)
t_values, U_crank_nicolson = solve_oscillator(Crank_Nicolson, U0, dt, t_max, linear_oscillator)

# Graficar los resultados
plt.figure(figsize=(12, 8))
plt.plot(t_values, U_euler[:, 0], label='Euler')
plt.plot(t_values, U_euler_inverso[:, 0], label='Euler Inverso')
plt.plot(t_values, U_crank_nicolson[:, 0], label='Crank-Nicolson')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.title('Linear Oscillator Solutions with Different Numerical Methods')
plt.show()

"""
Interpretación de los Resultados
Método de Euler:
    Si el paso de tiempo (dt) es grande, la solución numérica puede mostrar oscilaciones 
    crecientes y eventualmente divergir. Esto se debe a que el método de Euler es solo 
    condicionalmente estable y su región de estabilidad es pequeña.
Método de Euler Inverso:
    La solución numérica permanece estable incluso para pasos de tiempo más grandes. Sin embargo, 
    si el paso de tiempo es demasiado grande, la solución puede ser menos precisa debido a errores
    de truncamiento.

Interpretación de los Resultados

Método de Crank-Nicolson:
    La solución numérica es estable y más precisa que la del método de Euler inverso. Esto se
    debe a que el método de Crank-Nicolson es incondicionalmente estable y trata el tiempo de
    manera simétrica, reduciendo los errores de fase. 


Conclusión
    Las regiones de estabilidad absoluta te permiten entender cómo elegir el paso de tiempo
    adecuado para cada método numérico y cómo se comportan estos métodos en diferentes 
    situaciones. En general:

        -Métodos condicionalmente estables (como Euler) requieren pasos de tiempo pequeños 
        para mantener la estabilidad.
        -Métodos incondicionalmente estables (como Euler inverso y Crank-Nicolson) permiten
        pasos de tiempo más grandes, pero aún deben equilibrar la precisión y los errores de
        truncamiento.
"""