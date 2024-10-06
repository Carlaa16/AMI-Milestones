using LinearAlgebra
using Plots

# Definir el método de Euler
function Euler(U, dt, t, F)
    return U .+ dt .* F(U, t)
end

# Función de Kepler
function Kepler(U, t)
    x, y, dxdt, dydt = U
    d = (x^2 + y^2)^1.5
    return [dxdt, dydt, -x/d, -y/d]
end

# Método de punto fijo para Crank-Nicolson
function fixed_point_iteration(U_n, dt::Float64; tol=1e-6, max_iter=100)
    U_next = copy(U_n)  # Inicializamos U_{n+1} como U_n (primera aproximación)
    
    for _ in 1:max_iter
        U_next_old = copy(U_next)
        U_next .= U_n .+ (dt / 2) .* (Kepler(U_n, 0) .+ Kepler(U_next_old, 0))
        
        # Verificamos la convergencia
        if norm(U_next - U_next_old) < tol
            break
        end
    end

    return U_next
end

# Implementación de Runge-Kutta de cuarto orden
function RungeKutta(U, dt, t, F)
    k1 = F(U, t)
    k2 = F(U .+ dt/2 .* k1, t + dt/2)
    k3 = F(U .+ dt/2 .* k2, t + dt/2)
    k4 = F(U .+ dt .* k3, t + dt)
    return U .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

# Abstracción para Euler
function abstraction_for_F_and_Euler()
    U = [1.0, 0.0, 0.0, 1.0]  # Asegurar que U tenga tipo Float64
    N = 200
    x = zeros(Float64, N)
    y = zeros(Float64, N)
    t = zeros(Float64, N)
    
    x[1] = U[1]
    y[1] = U[2]
    t[1] = 0.0

    for i in 2:N
        dt = 0.1
        t[i] = dt * i
        U = Euler(U, dt, t, Kepler)
        x[i] = U[1]
        y[i] = U[2]
    end

    plot(x, y, title="Órbita Kepleriana usando Euler", xlabel="x", ylabel="y")
end

# Abstracción para Crank-Nicolson
function abstraction_for_F_and_Crank_Nicolson()
    U = [1.0, 0.0, 0.0, 1.0]  # Asegurar que U tenga tipo Float64
    N = 200
    x = zeros(Float64, N)
    y = zeros(Float64, N)
    t = zeros(Float64, N)
    
    x[1] = U[1]
    y[1] = U[2]
    t[1] = 0.0

    for i in 2:N
        dt = 0.1
        t[i] = dt * i
        U = fixed_point_iteration(U, dt)
        x[i] = U[1]
        y[i] = U[2]
    end

    plot(x, y, title="Órbita Kepleriana usando Crank-Nicolson", xlabel="x", ylabel="y")
end

# Abstracción para Runge-Kutta
function abstraction_for_F_and_RungeKutta()
    U = [1.0, 0.0, 0.0, 1.0]  # Asegurar que U tenga tipo Float64
    N = 200
    x = zeros(Float64, N)
    y = zeros(Float64, N)
    t = zeros(Float64, N)

    x[1] = U[1]
    y[1] = U[2]
    t[1] = 0.0

    for i in 2:N
        dt = 0.1
        t[i] = dt * i
        U = RungeKutta(U, dt, t[i], Kepler)
        x[i] = U[1]
        y[i] = U[2]
    end

    plot(x, y, title="Órbita Kepleriana usando Runge-Kutta", xlabel="x", ylabel="y")
end

# Main
function main() # Pongo en comentario las funciones que no se van a utilizar
   # abstraction_for_F_and_Euler()
    abstraction_for_F_and_Crank_Nicolson()
   # abstraction_for_F_and_RungeKutta()
end

main()
