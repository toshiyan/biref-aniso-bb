
# general
import numpy as np
import pickle
import subprocess
import sys
import os
import camb
from tqdm import tqdm
from scipy.integrate import simps
from scipy.special import spherical_jn as jl


# fixed values
ROOT = "/global/homes/t/toshiyan/Work/Ongoing/rotation/biref-aniso-bb/data/calEE/"


def set_ini_filename(k0,L0):
    return ROOT + "inifiles/run_params_k" + str(np.log10(k0))[:6] + "_L" + str(L0) + ".ini"


def set_calEE_filename(k0,L0):
    return ROOT + "aps/k" + str(np.log10(k0))[:6] + "_L" + str(L0)


def prep_inifile(k0,L0,verbose=False):
    filename = set_ini_filename(k0,L0)
    #os.system("cp class_acb/run_params_base.ini "+filename)
    #os.system("echo 'k0 = "+str(k0)+"' >> "+filename)
    #os.system("echo 'L0 = "+str(L0)+"' >> "+filename)
    #os.system("echo 'root = "+set_calEE_filename(k0,L0)+"' >> "+filename)
    #subprocess.run(["cp","class_acb/run_params_base.ini",filename])
    #subprocess.run(["echo","'k0 = "+str(k0)+"' >> "+filename])
    #subprocess.run(["echo","'L0 = "+str(L0)+"' >> "+filename])
    #'''
    f = open(filename, "w")
    f.write("k0 = "+str(k0)+"\n")
    f.write("L0 = "+str(L0)+"\n")
    f.write("output = tCl,pCl\n")
    f.write("modes = s\n")
    f.write("root = "+set_calEE_filename(k0,L0)+"\n")
    f.write("l_max_scalars = 5000\n")
    f.write("overwrite_root = yes\n")
    if verbose:
        f.write("input_verbose = 1\n")
        f.write("background_verbose = 1\n")
        f.write("thermodynamics_verbose = 1\n")
        f.write("perturbations_verbose = 1\n")
        f.write("transfer_verbose = 1\n")
        f.write("primordial_verbose = 1\n")
        f.write("harmonic_verbose = 1\n")
        f.write("fourier_verbose = 1\n")
        f.write("lensing_verbose = 1\n")
        f.write("distortions_verbose = 1\n")
        f.write("output_verbose = 1\n")
    f.close()
    #'''
    return filename


def prep_camb(lemax):
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lemax, lens_potential_accuracy=0);
    # for conformal time
    back = camb.get_background(pars)
    #calculate results for these parameters
    results = camb.get_results(pars)
    #get dictionary of CAMB power spectra
    EE = results.get_cmb_power_spectra(pars, CMB_unit=None, raw_cl=True)['unlensed_scalar'][:,1]
    BB = results.get_cmb_power_spectra(pars, CMB_unit=None, raw_cl=True)['lensed_scalar'][:,2]
    return EE, BB, back



def Delta_k_l(ell,ki,x_int,gvis,trans_x):
    
    kn = len(ki)
    ln = len(ell)

    Dkl = np.zeros((ln,kn),dtype=np.complex64)
    kx  = np.array([k*x_int for k in ki])
    gv  = np.array([ gvis for k in ki ]) 

    for l in tqdm(ell,desc='Delta_k_l'):
    
        if l<1: continue
        
        Dkl[l,:] = simps( jl(l,kx) * gv * trans_x, x_int )
    
    return Dkl


def calP_phi(k,A=1.,n=0.):
    
    return A*k**n


def compute_aps(ell,ki,Dkl0,Dkl1):
    
    ln, kn = np.shape(Dkl0)
    ln, kn = np.shape(Dkl1)
    
    cl = np.zeros((ln))

    for l in ell:
        cl[l] = 4.*np.pi * simps( Dkl0[l]*np.conjugate(Dkl1[l]) * calP_phi(ki) / ki , ki ).real
    
    return cl


def compute_claa(ell,lnkmin,lnkmax,lnkn,back,zmin=800,zmax=1200,tophat=False,transfunc=True):
    
    ki = np.logspace(lnkmin, lnkmax, num=lnkn)
    z_int = np.geomspace(zmin,zmax,100)
    x_int = back.comoving_radial_distance(z_int)
    t_int = back.conformal_time(z_int)
    
    if tophat: # tophat function
        gvis = (1./(x_int[-1]-x_int[0]))*(np.heaviside(x_int-x_int[0],0) + np.heaviside(x_int[-1]-x_int,0.) - 1.)
    else: # visibility function
        gvis = back.get_background_redshift_evolution(z_int, ['visibility'], format='array')[:,0]
        N = simps(gvis,x_int)
        gvis /= N
    
    if transfunc:
        trans = 3*np.array( [ jl(1,k*t_int)/(k*t_int) for k in ki] )
    else:
        trans = 1.

    Dkl_a = Delta_k_l(ell,ki,x_int,gvis,trans)

    return compute_aps(ell,ki,Dkl_a,Dkl_a)



def compute_calC(ks,Lmin=1,Lmax=100,lemin=2,lemax=6000):
    
    dks = np.zeros(len(ks))
    dks[:-1] = (ks[1:]-ks[:-1])*.5
    dks[-1] = dks[-2]

    le = np.linspace(2,lemax,lemax-1)
    
    calC = np.zeros((Lmax+1,lemax+1))
    for ki, k in enumerate(tqdm(ks)): 
        calEE = np.array( [ np.loadtxt(set_calEE_filename(k,L)+'_cl.dat',unpack=True,usecols=2) for L in range(Lmin,Lmax+1) ] )
        calC[Lmin:,lemin:] += 4*np.pi*calEE*(dks[ki]/k) * 2*np.pi/(le*(le+1))
    return calC



def compute_clbb(lbs,calC,Lmin=1,Lmax=100,lemin=2,lemax=6000):
    clBB = np.zeros(len(lbs))
    for i, lb in enumerate(tqdm(lbs)):
        w3j = np.load('/global/homes/t/toshiyan/scratch/rotation/w3j/wig3j_array_lb'+str(lb).zfill(4)+'.npz')    
        le = np.array( [ np.arange(max([np.abs(L-lb),lemin]),min([L+lb,lemax])+1) for L in range(Lmax+1) ] )    
        Wl = np.array( [ np.sum( (2.*le[L]+1.)*(2.*L+1.)*(1.+(-1.)**(lb+L+le[L]))*.5
                            * w3j[L-1,:len(le[L])]**2 * calC[L,min(le[L]):max(le[L])+1]
                           ) 
                    for L in np.arange(Lmin,Lmax+1)
                   ])    
        clBB[i] = np.sum(Wl)/np.pi
    return clBB
    