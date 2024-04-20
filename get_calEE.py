#!/usr/bin/env python3
# coding: utf-8

import numpy as np, tqdm, os, local, misctools, subprocess

ow = True
Lmax = 100
#Lmax = 2

ks = np.logspace(-5, 0, num=100)
#ks = np.logspace(-5, 0, num=50)
#ks = np.logspace(-5, 0, num=10)
Ls = np.linspace(1, Lmax, Lmax, dtype=int)

for L in tqdm.tqdm(Ls):
    for k in tqdm.tqdm(ks,leave=False):
        if misctools.check_path(local.set_calEE_filename(k,L)+'_cl.dat',overwrite=ow):
            continue
        fname = local.prep_inifile(k,L)
        subprocess.run(["./class_acb/class", fname])
        os.system("rm -rf "+fname)



