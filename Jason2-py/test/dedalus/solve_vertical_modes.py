# Find first 4 semi-diurnal modes

# Run with python3 in HPC

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from dedalus import public as de
import logging
logger = logging.getLogger(__name__)

def modes_from_N2(lat,N_array,d,tidal_periods):
    '''
    Finds first four modes of the semi-diurnal tide
    
    Inputs:
        - lat: Latitude in decimal degrees North
        - N_array: array of buoyancy frequency values at each depth in s^-2
        - d: array of depth in m
   	- tidal_periods (list) : periods of tidal wave in hours
     Outputs
        - first four modes in km
        - saves first 4 eigenvalues and  eigenvectors and depth as csv files
    '''
    # parameters
    strat = N_array[N_array != 0]
    depth = d[N_array != 0]
    H = depth[-1] 
    f = 2*7.2921e-5*np.sin(np.pi / 180 * lat) # coriolis parameter
    omgs = [2*np.pi/(tp * 3600) for tp in tidal_periods] #tidal frequency

    # number of modes for discretization
    nz = 128
    
    # build domain
    z_basis = de.Chebyshev('z', nz, interval=(H, -10))
    domain = de.Domain([z_basis], np.complex128)
    
    # non-constant coefficients
    N = domain.new_field(name='N')
    z = domain.grid(0)
    
    # interpolate N onto z
    N_interp = interpolate.interp1d(depth,strat)
    N['g'] = np.array(N_interp(z))
    
    # set up problem
    problem = de.EVP(domain, variables=['F', 'Fz'], eigenvalue='gam')
    problem.parameters['N'] = N
    problem.add_equation('dz(Fz/N**2) + gam*F = 0')
    problem.add_equation('Fz - dz(F) = 0')
    problem.add_bc('left(Fz) = 0')
    problem.add_bc('right(Fz) = 0')

    # set up solver
    solver = problem.build_solver()

    # solve problem
    solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)

    # sort eigenvalues
    gam = solver.eigenvalues
    gam[np.isnan(gam)] = np.inf
    idx = np.argsort(gam.real)
    
    # plot first four modes, print wavelengths
    modes = np.zeros((len(tidal_periods),4))
    eigenvectors = np.zeros((4,len(z))) 
    for n in range(4):
        solver.set_state(idx[n])
        F = solver.state['F']
        c = 1/np.sqrt(gam[idx[n]].real)
        for tp in range(len(tidal_periods)):
                k = np.sqrt(omgs[tp] ** 2 - f ** 2) / c
                modes[tp,n] = 1e-3 * 2 * np.pi / k
                print("tide {:d} mode {:d}: {:6.2f} km".format(tp,n, 1e-3*2*np.pi/k))
        eigenvectors[n,:] = F['g'].real
    e_vector = np.vstack((z,eigenvectors))
    np.savetxt('mode_z.csv',e_vector,delimiter=',')
    np.savetxt('modes.csv',modes,delimiter=',')    
    return modes
