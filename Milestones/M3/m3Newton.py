from numpy import array, linspace, zeros, size, shape, log10
from numpy.linalg import norm
from scipy.stats import linregress

def Richardson(U0, t1, F, ET, problema, orden):
    N = size(t1)
    t2 = linspace(0, t1[-1], 2*size(t1))
    Er = zeros((len(U0), N))
    
    U1 = problema(U0, t1, F, ET)
    U2 = problema(U0, t2, F, ET)
    
    for i in range(N):
        Er[:,i] = (U2[:,2*i] - U1[:,i]) / (1 - 1/2**orden)
    
    return Er

import numpy as np
from numpy.linalg import norm
from scipy.stats import linregress

def Convergencia(U0, t1, F, ET, problema, orden):
    Nt1 = len(t1)
    log_Er = np.zeros(orden)
    log_N = np.zeros(orden)
    
    U1 = problema(U0, t1, F, ET)
    
    for i in range(orden):
        Nt2 = 2 * Nt1
        t2 = np.linspace(0, t1[-1], Nt2)
        U2 = problema(U0, t2, F, ET)
        
        error = norm(U2[:,-1] - U1[:,-1])
        if error == 0:
            log_Er[i] = -np.inf
        else:
            log_Er[i] = np.log10(error)
        
        log_N[i] = np.log10(Nt2)
        t1 = t2
        U1 = U2
    
    # Filtrar valores -inf de log_Er y sus correspondientes en log_N
    valid_indices = ~np.isinf(log_Er)
    log_Er = log_Er[valid_indices]
    log_N = log_N[valid_indices]
    
    if len(log_N) < 2:
        raise ValueError("No hay suficientes puntos válidos para calcular la regresión lineal.")
    
    return log_Er, log_N, linregress(log_N, log_Er)

# --- Ejemplo de ejecución ---
import numpy as np

# Definimos una función `problema` simple como ejemplo
def problema(U0, t, F, ET):
    # Ejemplo: simulación de una función exponencial decreciente
    return np.array([[U0 * np.exp(-ET * t_i)] for t_i in t]).T

# Datos de entrada de ejemplo
U0 = np.array([1.0])  # Condición inicial
t1 = np.linspace(0, 1, 50)  # 50 puntos en el tiempo
F = None  # Placeholder (ya que F no se usa en este ejemplo)
ET = 0.1  # Parámetro de ejemplo
orden = 2  # Supón que es un método de segundo orden

# Llamada a la función Convergencia con todos los argumentos necesarios
log_Er, log_N, regression = Convergencia(U0, t1, F, ET, problema, orden)
Er = Richardson(U0, t1, F, ET, problema, orden)

print("Errores de Richardson:", Er)
print("Logaritmo de los errores:", log_Er)
print("Logaritmo del número de puntos:", log_N)
print("Resultado de la regresión (pendiente y coeficiente):", regression)
