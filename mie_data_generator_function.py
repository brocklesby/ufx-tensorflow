### Code to generate Mie Theta for the nano mie X-Ray scattering project ### 

# wsb version for playing with...
#imports
import math
import numpy as np
from scipy.special import jv, yv, jve, yve
import matplotlib.pyplot as plt
import pandas as pd
#import pickle 

def calculate_jn_yn(n, x):

    '''
    Returns jn and yn - both are functions of x.

        Parameters:

            n (int): Number of terms that are calculated. Calculated parameter set above 
            x (float): Caluculated parameter from above

        Returns:

            jj(array): __Needs explanation__
            y (array): __Needs explanation__
    '''
    #print(n,x)
    jj = np.zeros((int(n)+1, 1),dtype = 'complex_') # Setting up matrix of correct shape to hold values of j. X is 1D so fine to have value 1 in here.
    y = np.zeros(np.shape(jj),dtype = 'complex_') # Setting up matrix of size j, to hole values of y 

    for i in range(1, int(n)+1):

        jj[i-1] = np.sqrt(np.pi/(2*x)) * jv((i-1)+0.5, x)
        y[i-1] = np.sqrt(np.pi/(2*x)) * yv((i-1)+0.5, x)

    jj = jj[:-1]
    y = y[:-1]


    return jj, y 


# Calculate psi and xsi 

def calculate_psi_xsi(jj, y, n, x):

    '''
    Returns psi and xsi - Riccati-Bessel Function B&H p.101 - needed to calculate an and bn functions of x
    Also refrective index-independant

        Parameters:

            jj (array):  Bessel function
            y (array): bessel function
            n (int): Maximum number of terms to calculate over 
            x (float): product of k and radius

        Returns:

            psi (?): Riccati-Bessel function
            xsi (?): Riccati-Bessel function

    '''

    psi = np.zeros((int(n)+1, 1),dtype = 'complex_') # Setting up Ndim array
    xsi = np.zeros((int(n)+1, 1),dtype = 'complex_') # Setting up Ndim array 

    for i in range(0, int(n)):

        psi[i] = jj[i] * x 
        xsi[i] = np.multiply((jj[i] + 1j*y[i]), x)
    
    psi = psi[:-1]
    xsi = xsi[:-1]

    return psi, xsi


# Calculate D

def calculate_D(m, x, n):
#def calculate_D(m, x, n, n_factor):


    '''
    Calculate D, the log derivative of psi. This is D of mx, so is refractive index dependent 

        Parameters:

            m (int): Ratio of index 2 to index 1
            x (float): product of k and radius
            n (int): Maximum number of terms to calculate over 

        Returns:

            D (array): log derivative of psi

    '''

    mx = m*x 
    n_factor = 1.5 #fix this where we know it converges
    n = np.round(n_factor * n)  # this is where the extra D iterations come in...hope it just works
    D = np.zeros((int(n), 1), dtype = 'complex_')

#    for i in range(int(n)-1, 0, -1):
    for i in range(int(n)-1, 0, -1):

 #       D[i-2] = ((i-2)/mx) - 1.0/( (D[i-1]) + ((i-2)/mx) )
        D[i-1] = ((i)/mx) - 1.0/( (D[i]) + ((i)/mx) )
        
    #D = D[1:]
   
    return D


def calculate_an_bn(D, m, x, psi, xsi, n):

    '''
    Calculates an and bn. They are both column vectors 


        Parameters:
        
            D (array): Log derivative of mx
            m (int): Ratio of index 2 to index 1
            x (float): Calculated parameter 
            psi (?): __Needs explanation__
            xsi (?): __Needs explanation__
            n (int): Number of terms that are calculated. Calculated parameter set above 
    
        Returns:

            a (ndim array): scattering parameter
            b (ndim array): scattering parameter
    '''


    a = np.zeros((n, 1),dtype = 'complex_')
    b = np.zeros((n, 1),dtype = 'complex_')

    atop = np.zeros((int(n), 1),dtype = 'complex_')
    abottom = np.zeros((int(n), 1),dtype = 'complex_')
    btop = np.zeros((int(n), 1),dtype = 'complex_')
    bbottom = np.zeros((int(n), 1),dtype = 'complex_')


    for i in range(1, int(n)):
        
        atop = ((D[i]/m + i/x) * psi[i]) - psi[i-1]
        abottom = ((D[i]/m + i/x) * xsi[i]) - xsi[i-1]
        btop = ((m * D[i] + i/x) * psi[i]) - psi[i-1]
        bbottom = ((m * D[i] + i/x) * xsi[i]) - xsi[i-1]

        a[i] = atop/abottom
        b[i] = btop/bbottom

    return a, b 

# Calculate Qext

""" def calculate_Qext(n, a, b):

    '''
    Calculates C in order to calculate Qext.


        Parameters:
        
            a (ndim array): scattering parameter
            b (ndim array): scattering parameter
            n (int): Number of terms that are calculated. Calculated parameter set above 
    
        Returns:

            Qext (array): extinction coefficient

        '''


    C = np.zeros((int(n), 1), dtype = 'complex_')

    for i in range(1, int(n)):

        C[i] = (2*(i)+1) * np.real(a[i] + b[i])


    Qext = (2*np.pi/k**2) * C.sum() / (np.pi*np.array(r)**2)

    return Qext,C """



def calculate_Pi_tau(mu, n, theta_2D):

    '''
    Calculates Pi and tau
    See Bohren & Huffman eqn 4.46, p.94
    note that n starts from 0 here, so the python index is equal to n

        Parameters:
        
            mu (numpy array): cos of theta - theta can be vector for speed
            n (int): Number of terms that are calculated. Calculated parameter set above 
            theta_2D (numpy ndarray): Reshaped theta to become a 2D array for calcs
    
        Returns:

            Pi (numpy ndarray):  angular scattering function
            tau (numpy ndarray): angular scattering function

        '''


    p, q = np.shape(theta_2D) #Unpack shape output - expecting a ROW variable
    Pi = np.zeros((n, q), dtype='complex_') # Make a matrix of zeroes to hold Pi - added complex datatype

    tau = np.zeros(np.shape(Pi), dtype='complex_') # Matrix of zeros to hold tau 
 
    Pi[0] = 0 # Dont think this is needed but it's in original code 
    Pi[1] = 1 # Set 2nd row equal to 1

    #tau[2] = np.multiply(mu, (Pi[1] - Pi[0])) # think this is wrong maths - now fixed below 
    #tau[1] = np.multiply(mu, Pi[1]) - Pi[0] # think this is right
    #mu is an array of shape 256, 1
    foo1 = Pi[1]
    foo2 = np.multiply(mu, foo1)
    foo = foo2 - Pi[0] # break it down so I can see where it fails
    tau[1] = foo


    for i in range(2, n):

        Pi[i] = (( 2*i-1)/(i-1) * np.multiply(mu, Pi[i-1])) - ( i/(i-1) * Pi[i-2])
        tau[i] = (i) * np.multiply(mu, Pi[i]) - (i+1) * Pi[i-1]

    return Pi, tau

def calculate_S1_S2(theta_2D, n, a, Pi, b, tau):

    '''
    Calculates S1 and S2, the scattering parameters 


        Parameters:
        
            a (numpy ndarray): __Needs explanation__
            b (numpy ndarray): __Needs explanation__
            Pi (numpy ndarray): __Needs explanation__
            tau (numpy ndarray): __Needs explanation__
            n (int): Number of terms that are calculated. Calculated parameter set above 
            theta_2D (numpy ndarray): Reshaped theta to become a 2D array for calcs - shoul dbe a ROW variable
    
        Returns:

            S1 (float): Calculated scattering parameter 1
            S2 (float): Calculated scattering parameter 2

        '''


    p, q = np.shape(theta_2D) # Unpacking the shape of theta
    S1 = np.zeros((p, q), dtype='complex_') # Setting up a matrix of zeros
    S2 = np.zeros((p, q), dtype='complex_') # Setting up a matrix of zeros 
#error here due to using p not q, as theta2D has to be a row variable in other places, so p=1, q=len(th)
    S1setup = np.zeros((n, q), dtype='complex_')  # added dtype to prevent error
    S2setup = np.zeros((n, q), dtype='complex_')

    for i in range(1, n-1): #can use 1 to start as a0 = 

        S1setup[i] = ((2*i + 1)/(i*(i+1))) * ( a[i] * Pi[i] + b[i]*tau[i] ) 
        S2setup[i] = ((2*i + 1)/(i*(i+1))) * ( a[i] * tau[i] + b[i]*Pi[i]) 


    S1 = np.sum(S1setup, axis = 0)
    S2 = np.sum(S2setup, axis = 0)


    return S1, S2


def calculate_Iperp_Ipar(S1, S2):

    '''
    Calculates Iperp and Ipar, the intensities 


        Parameters:
        
            S1 (float): Calculated scattering parameter 1
            S2 (float): Calculated scattering parameter 2
    
        Returns:

            Iperp (float): Intensity polarised perpendicular to inut
            Ipar (float): Intensity polarised parallel to input

        '''

    Iperp = S2 * np.conj(S2)
    Ipar = S1 * np.conj(S1)

    return Iperp.real, Ipar.real


def add_to_dict(num1, num2, numbers, dictionary):
    # Create a tuple with the two numbers as its elements
    key = (num1, num2)
    
    # Add the key-value pair to the dictionary
    dictionary[key] = numbers

def generate_data(ind, radius, th_steps=100):
#def generate_data(ind, radius, n_factor, th_steps):
# this is where teh actual data generation is done...
    #data_steps = 100  #this defines the number of data points at different indices/radii
    #print('Entering gen_data')
    lambda_ = 632.8e-9 # wavelength 
    index1 = 1 # N- Surounding material
    #th_steps = 1000 # Approximate number of steps in angle 
    #th_steps = 100 # Approximate number of steps in angle
    verbose = 1 # Don't think this is needed in my implementation 

# calculated parameters used in the calculations

#theta = np.arange(-np.pi, np.pi, 2*np.pi/th_steps)
    theta_start = 0.03
    theta_end = 0.22
    theta = np.linspace(theta_start, theta_end, th_steps)
#theta = np.arange(0, np.pi, 2*np.pi/th_steps)
    theta_2D = theta.reshape(len(theta), 1) # Reshape theta to force it 2D
#k = 2 * np.pi * np.array(index1)/np.array(lambda_)
    k = 2 * np.pi * index1/lambda_ # wavelength is no longer an array

#n = np.round(x + 4*x**(1/3) + 2) # number of terms that we are calculating 

    mu = np.cos(theta) #theta can be a vector, for spee

#index_range = np.linspace(1.3,1.8, data_steps)
#complex_index_range = index_range+1j*1e-8

#radius_range = np.linspace(6e-6, 10e-6, data_steps)

#Iperp_dict_value_store = {}
#Ipar_dict_value_store = {}
#Itotal_dict_value_store = {}

#for i, ind in enumerate(complex_index_range):
    
  #  print(i)
    
 #   for radius in radius_range:
        
        
        # Constants
        
    index2 = ind
    r = radius
    x = 2 * np.pi * index1 * r /lambda_ # radius is no longer an array
    #print('x=',x)
    m = index2/index1 
    n = int(np.ceil(x + 4*x**(1/3) + 2)) # Number of terms as an int so we can use it as an index
    #print('n=',n)
    jj,y = calculate_jn_yn(n,x)
    psi,xsi = calculate_psi_xsi(jj,y, n,x)
    D = calculate_D(m,x,n)
    #D = calculate_D(m,x,n, n_factor)
    a,b = calculate_an_bn(D,m,x,psi,xsi,n)
    #Qext,C = calculate_Qext(n,a,b)
    Pi,tau = calculate_Pi_tau(mu,n,theta_2D)
    S1,S2 = calculate_S1_S2(theta_2D, n, a, Pi, b, tau)
    Iperp,Ipar = calculate_Iperp_Ipar(S1,S2)
    Itotal = Ipar + Iperp
        
    return(theta, Itotal)
  

def mie_theta(lambda_, theta, radius, index1, index2):
    '''
    input angle must be a row vector or it all goes wrong
    '''
    if theta.shape[0] != 1:
        raise Exception("Theta MUST be a row variable")
    #version of generate which looks like the old mie_theta, wiht more general variables. 
    k = 2 * np.pi * index1/lambda_ # wavelength is no longer an array
    mu = np.cos(theta) #theta can be a vector, for speed
    x = 2 * np.pi * index1 * radius /lambda_ # radius is no longer an array
    m = index2/index1 
    n = int(np.ceil(x + 4*x**(1/3) + 2)) # Number of terms as an int so we can use it as an index
    jj,y = calculate_jn_yn(n,x)
    psi,xsi = calculate_psi_xsi(jj,y, n,x)
    D = calculate_D(m,x,n)
    a,b = calculate_an_bn(D,m,x,psi,xsi,n)
    Pi,tau = calculate_Pi_tau(mu,n,theta)
    S1,S2 = calculate_S1_S2(theta, n, a, Pi, b, tau)
    Iperp,Ipar = calculate_Iperp_Ipar(S1,S2)
    Itotal = Ipar + Iperp
        
    return(Itotal)
  








