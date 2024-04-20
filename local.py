
# general
import numpy as np
import pickle
import subprocess
import sys
import os
import camb
import cmb
import emcee
from tqdm import tqdm
from scipy.integrate import simps
from scipy.special import spherical_jn as jl
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.stats import kde


#//// fixed values ////#
ROOT = "/global/homes/t/toshiyan/Work/Ongoing/rotation/biref-aniso-bb/data/calEE/"

# sptpol data
nfreq = 3
nbins = 7
ndata = nfreq*nbins
ical_90x90   =   367.669099005
ical_90x150  =  -367.669099005
ical_150x150 = 38911.2850797


def set_ini_filename(k0,L0):
    return ROOT + "inifiles/run_params_k" + str(np.log10(k0))[:6] + "_L" + str(L0) + ".ini"


def set_calEE_filename(k0,L0,aps='aps'):
    return ROOT + aps+"/k" + str(np.log10(k0))[:6] + "_L" + str(L0)


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


def prep_camb(lemax,sptpol=False):
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #Planck 2018 TT,TE,EE+lowE+lensing
    if sptpol:
        pars.set_cosmology(H0=67.84, ombh2=0.022294, omch2=0.11837, tau=0.067, mnu=0.06, omk=0.)
        pars.InitPower.set_params(As=np.exp(3.0659)*1e-10, ns=0.969, r=0)
    else:
        pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.12, mnu=0.06, omk=0, tau=0.0544)
        pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
    pars.set_for_lmax(lemax, lens_potential_accuracy=1);
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



def compute_calC(ks,Lmin=1,Lmax=100,lemin=2,lemax=6000,aps='aps'):
    
    dks = np.zeros(len(ks))
    dks[:-1] = (ks[1:]-ks[:-1])*.5
    dks[-1] = dks[-2]

    le = np.linspace(2,lemax,lemax-1)
    
    calC = np.zeros((Lmax+1,lemax+1))
    for ki, k in enumerate(tqdm(ks)): 
        calEE = np.array( [ np.loadtxt(set_calEE_filename(k,L,aps)+'_cl.dat',unpack=True,usecols=2) for L in range(Lmin,Lmax+1) ] )
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
    

#////////////////////#
# Constraint
#////////////////////#

def cl2cb(BBl,wl,nfreq,nbins):
    BBb = np.zeros((nfreq*nbins))
    n = 0
    for i in range(nfreq):
        for b in range(nbins):
            BBb[n] = np.sum(wl[i*nbins+b,:]*BBl)
            n += 1
    return BBb


def read_bp_window():
    # bandpower window function
    f = open('data/SPTpol_cosmomc/data/sptpol_500d_bb/sptpol_500d_BB_bpwfs.bin', 'r')
    l0, l1 = np.fromfile(f,dtype=np.int32,count=2)
    window = np.reshape(np.fromfile(f,dtype=np.float64),(ndata,l1-l0+1))
    f.close()
    l = np.linspace(l0,l1,l1-l0+1)
    return l, window


def read_bp_cov():
    # bandpower covariance matrix
    f = open('data/SPTpol_cosmomc/data/sptpol_500d_bb/sptpol_500d_BB_covariance.bin', 'r')
    cov = np.reshape(np.fromfile(f,dtype=np.float64),(ndata,ndata))
    f.close()
    return cov


def read_bp_freq():
    # bandpower (each freq)
    bc, bb_0, bb_1, bb_2 = np.loadtxt('data/SPTpol_cosmomc/data/sptpol_500d_bb/sptpol_500d_BB_bandpowers.txt',unpack=True,usecols=(0,5,4,3))
    return bc, np.concatenate((bb_0,bb_1,bb_2))


def prep_bb_theory(l,window,BB_lens=None):
    
    BB, bb = {}, {}
    
    # lensing
    if BB_lens is None:
        BB_lens = prep_camb(int(np.max(l)))[1]
    
    BB['lens'] = (l*(l+1)*cmb.Tcmb**2/np.pi/2.)*BB_lens[int(np.min(l)):int(np.max(l))+1]

    # biref
    L, BB_arot = np.loadtxt('data/clbb.dat',unpack=True)
    BB['arot'] = spline( L, BB_arot*cmb.Tcmb**2*L*(L+1)/(2.*np.pi) )(l)
    
    # dust FGs
    BB['dust'] = (0.0094)*(l/80.)**(-0.58)
    
    # poisson FGs
    BB['pois'] = (l/3000.)**2

    # compute bandpower
    for t in ['lens','arot','dust','pois']:
        bb[t] = cl2cb(BB[t],window,nfreq,nbins)

    # correct scaling for dust
    bb['dust'][1*nbins:2*nbins] *= dust_scaling(96.2,149.5)
    bb['dust'][2*nbins:3*nbins] *= dust_scaling(96.2,96.2)

    return BB, bb


def dust_scaling(nu0,nu1,beta=1.59,Tdust=19.6):

    return ((nu0*nu1)/(150.*150.))**beta * Bnu(nu0,150.,Tdust)*Bnu(nu1,150.,Tdust) / (dBdT(nu0,150.)*dBdT(nu1,150.))


def dBdT(nu, nu0):
    x0 = nu0 / 56.78
    dBdT0 = x0**4 * np.exp(x0) / (np.exp(x0) - 1)**2
    x = nu / 56.78
    dBdT = x**4 * np.exp(x) / (np.exp(x) - 1)**2 / dBdT0

    return dBdT


def Bnu(nu, nu0, T, hk = 4.799237e-2):
    
    return (nu/nu0)**3 * (np.exp(hk*nu0/T)-1)/(np.exp(hk*nu/T)-1)


def lnL(params,bb,bb_data,cov,berr,n=1,arot=True,sAlens=0.025,mAdust=0.0094):
    
    if arot: 
        p0 = 3
        Acb, Alens, Adust = params[:p0]
    else:
        p0 = 2
        Alens, Adust = params[:p0]
        Acb = 0.
    
    Ap = params[p0:p0+n]
    if n==1: 
        cal = params[p0+n:p0+n+1]
        beam = params[p0+n+1:]
    else: 
        cal = params[p0+n:p0+n+2]
        beam = params[p0+n+2:]
    
    if all( 0<=x for x in params[:p0]) and all( 0<=x<=10 for x in Ap) and all( 0<=x for x in cal):

        bb_fit = np.zeros(ndata)
        for nu in range(n):
            if nu==0: c = cal[0]**2
            if nu==1: c = cal[0]*cal[1]
            if nu==2: c = cal[1]**2
            for bi in range(nu*nbins,(nu+1)*nbins):
                bb_fit[bi] = ( bb['arot'][bi]*Acb + bb['lens'][bi]*Alens + bb['dust'][bi]*Adust + bb['pois'][bi]*Ap[nu] ) / c

        for bi in range(7):
            bb_fit *= (1.+berr[:,bi]*beam[bi])

        diff = (bb_fit-bb_data)[:n*nbins]
        icov = np.linalg.inv(cov[:n*nbins,:n*nbins])
    
        # base likelihood
        lnL  = -0.5*np.sum( np.dot(diff,np.dot(icov,diff)) )
        
        # add priors
        if arot:
            lnL += -0.5*(Alens-1.)**2/sAlens**2 
        
        lnL += - 0.5*(Adust-mAdust)**2/0.0021**2 - 0.5*np.sum(beam**2)
        
        if n==1:
            lnL += -0.5*ical_150x150*(cal[0]-1.)**2
        else:
            lnL += -0.5*( ical_150x150*(cal[0]-1)**2 + 2*ical_90x150*(cal[0]-1)*(cal[1]-1) + ical_90x90*(cal[1]-1)**2 )

        return lnL
    
    else:
        
        return -np.inf
    
    
def lnL_set_params(n=1,arot=False,nwalkers=500,sAlens=0.5,mAdust=0.0094):

    # Acb, Alens, Adust
    sigma0 = np.array([1e-4,sAlens,0.005])
    pos0   = np.array([0.,1.,mAdust])
    
    if not arot:
        sigma0 = sigma0[1:]
        pos0   = pos0[1:]
    
    # poisson
    sigma_p = np.array([0.01,0.1,0.1])[:n]
    pos_p   = np.array([0.01,0.1,0.1])[:n]

    # cal
    if n == 1:
        sigma_c = 0.1*np.ones(1)
        pos_c   = np.ones(1)
    else:
        sigma_c = 0.1*np.ones(2)
        pos_c   = np.ones(2)

    ndim = len(sigma0) + len(sigma_p) + len(sigma_c) + 7
    sigma = np.concatenate( ( sigma0, sigma_p, sigma_c, np.ones(7) ) )
    pos = np.concatenate( ( pos0, pos_p, pos_c, np.zeros(7) ) ) + sigma[None,:] * np.random.randn(nwalkers, ndim)
    
    return ndim, nwalkers, sigma, pos
    

def run_mcmc(bb,bb_data,cov,berr,n=3,arot=True,nwalkers=500,sAlens=0.5,mAdust=0.0094,steps=5000,discard=100,thin=20):
    
    ndim, nwalkers, sigma, pos = lnL_set_params(n=n,arot=arot,nwalkers=nwalkers,sAlens=sAlens,mAdust=mAdust)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        lnL,
        args=[bb,bb_data,cov,berr,n,arot,sAlens,mAdust],
    )
    sampler.run_mcmc(pos, steps, progress=True)
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # Post-processing: Remove unphysical values
    valid_indices = np.where((flat_samples[:, 0] >= 0) & (flat_samples[:, 1] >= 0))[0]
    
    return ndim, flat_samples[valid_indices]

    
def highest_posterior_density_interval(samples, credible_interval=0.95, num_points=1000):
    # Estimate the kernel density of the samples
    kernel = kde.gaussian_kde(samples)
    
    # Generate a range of values for the posterior density
    x_values = np.linspace(np.min(samples), np.max(samples), num_points)
    
    # Compute the posterior density at each point
    posterior_density = kernel(x_values)
    
    # Sort the values and compute the cumulative distribution function (CDF)
    sorted_indices = np.argsort(posterior_density)[::-1]
    cdf = np.cumsum(posterior_density[sorted_indices]) / np.sum(posterior_density)
    
    # Find the indices corresponding to the desired credible interval
    lower_index = np.argmax(cdf >= (1 - credible_interval))
    upper_index = np.argmax(cdf >= credible_interval)
    
    # Extract the values corresponding to the HPD interval
    hpd_interval = x_values[sorted_indices[lower_index:upper_index+1]]
    
    return hpd_interval


def compute_hpd(samples):
    
    hpd_interval_68 = highest_posterior_density_interval(samples, credible_interval=0.6827)
    hpd_interval_95 = highest_posterior_density_interval(samples, credible_interval=0.9545)
    hpd_interval_99 = highest_posterior_density_interval(samples, credible_interval=0.9973)
    
    sigma_1 = np.array( [ np.min(hpd_interval_68),np.max(hpd_interval_68) ] )
    sigma_2 = np.array( [ np.min(hpd_interval_95),np.max(hpd_interval_95) ] )
    sigma_3 = np.array( [ np.min(hpd_interval_99),np.max(hpd_interval_99) ] )
    
    mean = np.mean(samples)
    
    print(mean,sigma_1-mean,sigma_2-mean)
    
    return mean, sigma_1, sigma_2, sigma_3, samples

