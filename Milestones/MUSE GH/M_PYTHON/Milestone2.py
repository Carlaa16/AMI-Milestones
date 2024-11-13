from ODES.Cauchy_problem import Cauchy_problem 
from ODES.Temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, Embedded_RK 
from Physics.Orbits import Kepler 
import matplotlib.pyplot as plt
from numpy import array 

 
from numpy import linspace


def Simulation(tf, N, U0): 
   
    """ 
    This is a Simulation code to integrate Cauchy problems with some numerical scheme.

    The objective of this Milestone is to create different functions or abstractions 
    by means of functional programming or function composition. 
    Namely, the following modules are created to allow this functional composition:
            1) ODES.Cauhy_problem
            2) ODES.Temporal_schemes
            3) Physics.Orbits
    
    The idea is to create a Cauchy problem abstraction to integrate different physical 
    problems with different temporal schemes.
    
    Different abstractions : 
                     1) dU/dt = F(U, t) 
                     2) Temporal scheme to integrate one step 
                     3) Cauchy problem to perform different steps 
                         
     """


    t = linspace(0, tf, N)
    schemes = [  (Euler, None, None ),  (Embedded_RK, 2, 1e-1), (Embedded_RK, 8, 1e-1)  ]

    
    for (method, order, eps)  in schemes:

       if order != None:  
          U =  Cauchy_problem( Kepler, t, U0, method, q=order, Tolerance=eps ) 
       else: 
          U =  Cauchy_problem( Kepler, t, U0, method ) 

       plt.axes().set_aspect('equal')
       plt.plot( U[:,0] , U[:,1], ".")
       plt.show()

if __name__ == "__main__":

  Simulation(100, 100, array( [ 1., 0., 0., 1. ] ) )

  """"
  Si compilaras manualmente el archivo de Cauchy_problem.py antes de ejecutar tu código,
  el archivo .pyc aún se crearía si Python no encuentra un archivo compilado válido en el
  directorio __pycache__. Sin embargo, el propósito del archivo .pyc es acelerar la carga
  de módulos. Python siempre intenta crear el archivo .pyc la primera vez que se ejecuta
  un script si no existe o si el archivo fuente (.py) ha sido modificado.
  """
  