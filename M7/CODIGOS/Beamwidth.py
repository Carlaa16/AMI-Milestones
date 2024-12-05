import numpy as np
from scipy.optimize import fsolve

# Constantes
R = 6371  # Radio de la Tierra en km
h = 850   # Altitud del satélite en km

def equation(epsilon_min, theta, R, h):
    """
    Ecuación que relaciona la elevación mínima con el ángulo (theta).
    """
    theta_rad = np.radians(theta)
    epsilon_rad = np.radians(epsilon_min)
    
    return (
        (h**2) * (np.cos(epsilon_rad)**2) 
        - 2 * ((R + h)**2) * (np.sin(theta_rad / 2 + epsilon_rad)**2) * (1 - np.cos(theta_rad))
    )

def calculate_elevation(theta, R, h):
    """
    Calcula la elevación mínima para un ángulo específico.
    """
    epsilon_min = fsolve(equation, x0=10, args=(theta, R, h))[0]  # Resolver la ecuación
    return epsilon_min

# Introduce el valor del ángulo (Theta)
theta = float(input("Introduce el valor del ángulo Theta (en grados): "))

# Calcular la elevación mínima
elevation_min = calculate_elevation(theta, R, h)

# Mostrar el resultado
print(f"Para un ángulo Theta de {theta:.2f}°, la elevación mínima es de {elevation_min:.2f}°.")
