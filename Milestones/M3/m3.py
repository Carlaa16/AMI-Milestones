from numpy import zeros, float64, array, linspace
import matplotlib.pyplot as plt

import numpy as np

def euler_method(f, y0, a, b, N):
    h = (b - a) / N
    t = a
    y = y0
    for _ in range(N):
        y += h * f(t, y)
        t += h
    return y

def inverse_euler_method(f, y0, a, b, N):
    h = (b - a) / N
    t = a
    y = y0
    for _ in range(N):
        y_next = y + h * f(t + h, y + h * f(t, y))
        y = y_next
        t += h
    return y

def crank_nicolson_method(f, y0, a, b, N):
    h = (b - a) / N
    t = a
    y = y0
    for _ in range(N):
        y_next = y + h/2 * (f(t, y) + f(t + h, y + h * f(t, y)))
        y = y_next
        t += h
    return y

def runge_kutta_4th_order(f, y0, a, b, N):
    h = (b - a) / N
    t = a
    y = y0
    for _ in range(N):
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
    return y

def richardson_extrapolation(f, y0, a, b, N, method):
    if method == 'Euler':
        method_func = euler_method
    elif method == 'Inverse Euler':
        method_func = inverse_euler_method
    elif method == 'Crank Nicolson':
        method_func = crank_nicolson_method
    elif method == 'Runge Kutta':
        method_func = runge_kutta_4th_order
    else:
        raise ValueError("Unknown method")

    y_h = method_func(f, y0, a, b, N)
    y_h2 = method_func(f, y0, a, b, 2*N)

    error_estimate = (y_h2 - y_h) / (2**2 - 1)  # Assuming second-order accuracy
    return error_estimate

# Example usage:
# Define the differential equation dy/dt = f(t, y)
def f(t, y):
    return -2 * y + t

# Initial condition
y0 = 1

# Time interval
a = 0
b = 1

# Number of steps
N = 10

# Method
method = 'Runge Kutta'

# Calculate error estimate
error = richardson_extrapolation(f, y0, a, b, N, method)
print("Error estimate:", error)