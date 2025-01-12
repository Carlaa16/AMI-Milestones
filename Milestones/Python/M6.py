from numpy import arange, array, log10, zeros, vstack, ones, linspace, transpose, reshape, sqrt, round, pi, cos, sin, meshgrid, dot
from numpy.linalg import norm, lstsq
from scipy.optimize import fsolve, root, newton
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


################################################# FUNCIONES #######################################################

def Cauchy_problem(t, temporal_scheme, f, U0):
    U = array (zeros((len(U0),len(t))))
    U[:,0] = U0
    for ii in range(0, len(t) - 1):
        U[:,ii+1] = temporal_scheme(U[:,ii], t[ii], t[ii+1], f)
    return U

# Funcion para obtener la matriz de Butcher para diferentes órdenes de Runge Kutta
def obtener_array_Butcher(q): 
    N_stages = {2:2, 3:4, 8:13}

    N =  N_stages[q]
    a = zeros((N, N))
    b = zeros((N))
    bs = zeros((N))
    c = zeros((N)) 
    
    if q==2:
        a[0,:] = [0, 0]
        a[1,:] = [1, 0]

        b[:] = [1/2, 1/2]

        bs[:] = [1, 0]

        c[:]  = [0, 1]

    elif q==3:
        a[0,:] = [0, 0, 0, 0]
        a[1,:] = [1/2, 0, 0, 0]
        a[2,:] = [0, 3/4, 0, 0]
        a[3,:] = [2/9, 1/3, 4/9, 0]

        b[:]  = [2/9, 1/3, 4/9, 0]

        bs[:] = [7/24, 1/4, 1/3, 1/8]

        c[:] = [0, 1/2, 3/4, 1]
    
    elif q==8:
        a[0,:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
        a[1,:] = [2/27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
        a[2,:] = [1/36, 1/12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
        a[3,:] = [1/24, 0, 1/8 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
        a[4,:] = [5/12, 0, -25/16, 25/16, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a[5,:] = [1/20, 0, 0, 1/4, 1/5, 0, 0, 0, 0, 0, 0, 0, 0] 
        a[6,:] = [-25/108, 0, 0, 125/108, -65/27, 125/54, 0, 0, 0, 0, 0, 0, 0] 
        a[7,:] = [31/300, 0, 0, 0, 61/225, -2/9, 13/900, 0, 0, 0, 0, 0, 0] 
        a[8,:] = [2, 0, 0, -53/6, 704/45, -107/9, 67/90, 3, 0, 0, 0, 0, 0] 
        a[9,:] = [-91/108, 0, 0, 23/108, -976/135, 311/54, -19/60, 17/6, -1/12, 0, 0, 0, 0] 
        a[10,:] = [2383/4100, 0, 0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41, 0, 0, 0] 
        a[11,:] = [3/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0, 0, 0]
        a[12,:] = [-1777/4100, 0, 0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 19/41, 0, 1, 0]
    
        b[:]  = [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 41/840, 0, 0]

        bs[:] = [0, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 0, 41/840, 41/840]   

        c[:] = [0, 2/27, 1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3 , 1/3, 1, 0, 1]  
    
    else:
        print("Butcher array  not avialale for order =", q)
        exit()

    return a, b, bs, c 

# Calculo de los valores de las k
def RK_k_calculation(f, U0, t0, dt, a, c):
    k = zeros((len(c), len(U0)))
    for i in range(len(c)):
        Up = U0 + dt * dot(a[i, :], k)
        k[i,:] = f(t0 + c[i]*dt, Up)
    return k

# RK EMBEBIDO
def Embedded_RK(U0, t0, tf, f, q, Tolerance): 
    dt = tf - t0

    a, b, bs, c = obtener_array_Butcher(q)
    k = RK_k_calculation(f, U0, t0, dt, a, c)

    # Error = dot(b-bs, k)
    Error = dt * dot(bs-b, k)
    dt_min = min(dt, dt * (Tolerance / norm(Error))**(1/q))
    N = int(dt/dt_min) + 1
    h = dt / N
    Uh = U0[:]

    for i in range(0, N):
        k = RK_k_calculation(f, Uh, t0 + h*i, h, a, c)
        Uh += h * dot(b, k)

    return Uh

# Definicion de una funcion adicional para pasar los valores definidos de q y tol a la funcion Embedded_RK
def RK_emb(U0, t0, tf, f):
    return Embedded_RK(U0, t0, tf, f, q, tol)


# Problema de los 3 cuerpos circular restringido
def N3_body_restricted(t, U, m1, m2, r12):
    x, y, vx, vy = U[0], U[1], U[2], U[3]

    G = 6.6743e-20 # [N km^2 kg^-2]

    omega = sqrt(G*(m1+m2)/(r12**3))
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    mu1 = G * m1
    mu2 = G * m2

    x1 = -r12*pi2
    x2 = r12*pi1

    r1 = sqrt((x - x1)**2 + y**2)
    r2 = sqrt((x - x2)**2 + y**2)
    
    dotdotx = omega**2 * x + 2*omega*vy - (mu1/r1**3) * (x + pi2*r12) - (mu2/r2**3) * (x - pi1*r12)
    dotdoty = omega**2 * y - 2*omega*vx - (mu1/r1**3) * y - (mu2/r2**3) * y

    return array([vx, vy, dotdotx, dotdoty])

# Definicion de una funcion adicional para pasar los valores definidos de m1, m2, pos1 y pos2 a la funcion N3_body_restricted
def CR3BP(t, U):
    return N3_body_restricted(t, U, m1, m2, r12)

# Funcion para obtener la posicion de los planetas en eje x, en ejes centrados en el baricentro del sistema compuesto por m1 y m2
def obtener_pos_planetas(m1, m2, r12):
    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)
    pos1 = -r12*pi2
    pos2 = r12*pi1
    return(pos1, pos2)

def CR3BP_Lagrange(x):
    U = array([x[0], x[1], 0, 0])
    Sol = N3_body_restricted(0, U, m1, m2, r12)
    return array([Sol[0], Sol[1]])

def Stability_Lagrange(x, y, m1, m2, r12):
    # x = linspace(lims_malla[0,0], lims_malla[0,1], 100)
    # y = linspace(lims_malla[1,0], lims_malla[1,1], 100)

    pi1 = m1 / (m1 + m2)
    pi2 = m2 / (m1 + m2)

    U = zeros([len(x), len(y)])
    for ii in range(0, len(x)):
        for jj in range(0, len(y)):
            sigma = sqrt((x[ii]/r12 + pi2)**2 + (y[jj]/r12)**2)
            psi = sqrt((x[ii]/r12 - pi1)**2 + (y[jj]/r12)**2)
            U[ii,jj] = -pi1/sigma - pi2/psi - 1/2 * (pi1*sigma**2 + pi2*psi**2)

    return(U)


################################################### CÓDIGO ########################################################
# Definicion del orden y tolerancia del RK embebido
q = 8
tol = 1e-12

# Definicion de los parametros del problema de los 3 cuerpos circular restringido. Se incluyen los datos Sol-Tierra y Tierra-Luna
m1 = 3.955e30 # Masa del Sol (kg)
m2 = 5.972e24 # Masa de la Tierra (kg)
r12 = 149597870 # Distancia Sol-Tierra [km]
R_1 = 696340 # Radio del Sol, para la representacion [km]
R_2 = 6378 # Radio de la Tierra, para la representacion [km]

# m1 = 5.972e24 # Masa de la Tierra (kg)
# m2 = 7.349e22 # Masa de la Luna (kg)
# r12 = 384403 # Distancia Tierra-Luna [km]
# R_1 = 6378 # Radio de la Tierra, para la representacion [km]
# R_2 = 1737 # Radio de la Luna, para la representacion [km]

# m1 = 3.955e25 # Masa del Sol (kg)
# m2 = 5.972e24 # Masa de la Tierra (kg)
# r12 = 1495978 # Distancia Sol-Tierra [km]
# R_1 = 69634 # Radio del Sol, para la representacion [km]
# R_2 = 6378 # Radio de la Tierra, para la representacion [km]

# Definicion de los parametros de integracion
N = 10000
t0 = 0
tf = 10000
t = linspace(t0, tf, N+1)

# Resolucion del problema de Cauchy_problem
pos1, pos2 = obtener_pos_planetas(m1, m2, r12)
U0 = [pos2 + 6378 + 500, 0, 0, 11]
U = Cauchy_problem(t, RK_emb, CR3BP, U0)

# Calculo de puntos criticos
initial_guess = array([[0.8*r12, 0], [1.2*r12, 0], [-1*r12, 0], [0.5*r12, 0.5*r12], [0.5*r12, -0.5*r12]])
sol_Lagrange = zeros((2,5))
for ii in range(0,5):
    sol_Lagrange[:,ii] = newton(CR3BP_Lagrange, initial_guess[ii])

for ii in range(0,5):
    for jj in range(0,2):
        if abs(sol_Lagrange[jj,ii]) < 1e-4:
            sol_Lagrange[jj,ii] = 0

# print(sol_Lagrange)
            
# Cálculo de la estabilidad de los puntos de Lagrange
x = linspace(-r12*2, r12*2, 1000)
y = linspace(-r12*2, r12*2, 1000)
Potencial = Stability_Lagrange(x, y, m1, m2, r12)



################################################# GRÁFICAS #######################################################
# Representacion de resultados:
fig, ax = plt.subplots()
# - Representacion de las masas (se representan a escala real, por lo que es necesario ampliar en el punto entre L1 y L2 para distinguir la masa 2 si esta es mucho mas pequeña que la 1)
m1_circle = plt.Circle((pos1, 0), R_1, color='yellow', label='M1')
m2_circle = plt.Circle((pos2, 0), R_2, color='blue', label='M2')
ax.add_patch(m1_circle)
ax.add_patch(m2_circle)
# - Representacion de las trayectorias de las masas alrededor del baricentro
theta_m = linspace(0, 2*pi, 1000)
ax.plot(abs(pos1)*cos(theta_m), abs(pos1)*sin(theta_m), color='orange', linewidth=0.25)
ax.plot(abs(pos2)*cos(theta_m), abs(pos2)*sin(theta_m), color='blue', linewidth=0.25)
# - Representacion de la trayectoria
ax.plot(U[0,:],U[1,:], label='Trayectoria')
# - Representacion de los puntos de Lagrange
cmap = plt.get_cmap('Greens')
for ii in range(0, 5):
    color = cmap((8-ii) / 8)
    ax.plot(sol_Lagrange[0, ii], sol_Lagrange[1, ii], marker='o', color=color)
    ax.text(sol_Lagrange[0, ii], sol_Lagrange[1, ii], f'L{ii+1}', color='black', fontsize=10, ha='right', va='bottom')

ax.legend()
plt.axis('equal')
plt.show()
# - Representacion del potencial gravitatorio (estabilidad de los puntos de Lagrange)
x_mesh, y_mesh = meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_mesh, y_mesh, Potencial, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potencial gravitatorio')
ax.set_title('Estabilidad de los puntos de Lagrange')
plt.show()