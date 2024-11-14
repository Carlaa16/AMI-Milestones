import matplotlib.pyplot as plt
import numpy as np
from numpy import array, zeros, linspace, reshape

V = array([1,2,3,4]) # Este V apunta al 1,2,3
pV = V # Alias
pV[0] = 4 # Estoy cambiando en el alias el primer termino por 4
print (V) # Imprime 4,2,3
print(id(V), id(pV)) # Imprime la dirección de memoria de V

U = V.copy() # Hago una copia de V
U[0] = 18 # Cambio el primer termino de U

print (V) # Imprime 4,2,3
print(U)

print(id(V), id(pV), id(U)) # Imprime la dirección de memoria de U, que veremos que no es la misma que V

pU = reshape(U, (2,2)) # Cambio la forma de U a una matriz de 2x2
# reshape es un alias (está definido asi en numpy)
pU[0,0] = 8 # Cambio el primer termino de la matriz pU
print(U) 