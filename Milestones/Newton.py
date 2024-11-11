from numpy import array, zeros, linspace, exp, concatenate
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Función el punto y la derivada
def Newton (F, x0, Fp, tolerancia = 1e-8):

    xn = x0
    Error = tolerancia + 1
    while Error > tolerancia:

        xn1 = xn - F(xn) / F(xn)/Fp(xn)
        Error = abs(xn1 - xn)
        print("xn = ", xn, "xn1 = ", xn1 - xn)
        xn = xn1

    return xn

#FUnción que newton aproximará
def G(x):

    return exp(x) - 2 * x - 2

def Gp(x):

    return exp(x) - 2

def particion(a, b, N):

    t = zeros(N + 1)

    for i in range (0, N+1):
        t[i] = a + (b - a) / N * i

    return t

#Solucion = Newton(F = G, x0 = 0.5, Fp = Gp)

x = particion(a = - 2, b = 2, N = 100)
y = G(x)

plt.plot(x, y)
plt.show()
