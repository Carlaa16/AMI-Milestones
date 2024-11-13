 
from numpy import array, zeros, linspace
import numpy as np
import matplotlib.pyplot as plt

from miscellaneous import decorators 

#EULER

@decorators.profiling
def abstraction_for_F_and_Euler(): 

    U = array( [ 1, 0, 0, 1 ])
    
    N = 200 
    x = array( zeros(N) )
    y = array( zeros(N) )
    t = array( zeros(N) )
    x[0] = U[0] 
    y[0] = U[1]
    t[0] = 0 
    
    for i in range(1, N): 

      dt = 0.1 
      t[i] = dt*i
      U = Euler(U, dt, t, Kepler)
      x[i] = U[0] 
      y[i] = U[1]
    
    plt.plot(x, y)
    plt.title("Órbita Kepleriana usando Euler")
    plt.show()

def Kepler(U, t): 

    x = U[0]; y = U[1]; dxdt = U[2]; dydt = U[3]
    d = ( x**2  +y**2 )**1.5

    return  array( [ dxdt, dydt, -x/d, -y/d ] ) 

def Euler(U, dt, t, F): 

    return U + dt * F(U, t)


  # CRANK-NICOLSON

# Método de punto fijo para aproximar U_{n+1}
def fixed_point_iteration(U_n, dt, tol=1e-6, max_iter=100):
    U_next = U_n.copy()  # Inicializamos U_{n+1} como U_n (primera aproximación)
    
    for _ in range(max_iter):
        U_next_old = U_next.copy()
        # Crank-Nicolson: U_{n+1} = U_n + (dt / 2) * (F(U_n) + F(U_{n+1}))
        U_next = U_n + (dt / 2) * (Kepler(U_n, 0) + Kepler(U_next_old, 0))
        
        # Verificamos la convergencia
        if np.linalg.norm(U_next - U_next_old) < tol:
            break

    return U_next

# Implementación de Crank-Nicolson
@decorators.profiling
def abstraction_for_F_and_Crank_Nicolson(): 

    U = array([1, 0, 0, 1])
    
    N = 200 
    x = array(zeros(N))
    y = array(zeros(N))
    t = array(zeros(N))
    x[0] = U[0]
    y[0] = U[1]
    t[0] = 0 
    
    for i in range(1, N): 
        dt = 0.1
        t[i] = dt * i
        U = fixed_point_iteration(U, dt)
        x[i] = U[0]
        y[i] = U[1]
    
    plt.plot(x, y)
    plt.title("Órbita Kepleriana usando Crank-Nicolson")
    plt.show()

# RUNGE-KUTTA DE CUARTO ORDEN (RK4)

def RungeKutta(U, dt, t, F):
    k1 = F(U, t)
    k2 = F(U + dt/2 * k1, t + dt/2)
    k3 = F(U + dt/2 * k2, t + dt/2)
    k4 = F(U + dt * k3, t + dt)
    return U + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def abstraction_for_F_and_RungeKutta(): 

    U = array([1, 0, 0, 1])
    
    N = 200 
    x = zeros(N)
    y = zeros(N)
    t = zeros(N)
    x[0] = U[0]
    y[0] = U[1]
    t[0] = 0 
    
    for i in range(1, N): 
        dt = 0.1
        t[i] = dt * i
        U = RungeKutta(U, dt, t[i], Kepler)
        x[i] = U[0]
        y[i] = U[1]
    
    plt.plot(x, y)
    plt.title("Órbita Kepleriana usando Runge-Kutta 4")
    plt.show()

if __name__ == "__main__":
  
   abstraction_for_F_and_Euler()
   abstraction_for_F_and_Crank_Nicolson()
   abstraction_for_F_and_RungeKutta()
  