using NonlinearSolve
using LinearAlgebra

# Primera parte del código (sin parámetros)
g(u, p) = u^2 - 2.0  # Añadido el parámetro `p`, aunque no se use
u0 = 1.0

problem1 = NonlinearProblem(g, u0)
solution1 = solve(problem1)

println("Solution to first problem: ", solution1.u)

# Segunda parte del código (con parámetros)
f(u, p) = u .* u .- p  # Asegúrate de usar la operación element-wise `.*` con vectores

u0 = [1.0, 1.0]  # Condición inicial
p = 2.0  # Parámetro

problem2 = NonlinearProblem(f, u0, p)  # Definir el problema con parámetros
solution2 = solve(problem2)

println("Solution to second problem: ", solution2.u)
