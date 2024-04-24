This repository contains tools to compute the B-mode power spectrum induced by time-varying anisotropic cosmic birefringence. 
Details of the B-mode power spectrum formalism are described in Namikawa (2024) (https://inspirehep.net/literature/2779565). 

The following explains how to use this tool. 

Install 
---
1. Please first compile the modified CLASS code at `class_acb/`
   The modified CLASS code will compute the quantity, ${\cal C}^{EE}_{l,L}(k)$. 
3. Next you need to create a folder as
   ```
   mkdir -p data/calEE/aps data/calEE/inifiles
   ```
   The modified CLASS code will save ${\cal C}^{EE}_{l,L}(k)$ for each $L$ and $k$ at `data/calEE/aps` and also save parameter files at `data/calEE/inifiles`. 

Running the code
---
1. First prepare ${\cal C}^{EE}_{l,L}(k)$ by running the python script
   ```
   python get_calEE.py
   ```
3. Next prepare the Wigner 3y by running `get_w3jm.ipynb`. Please specify a directory to save the file. 
4. Finally, you obtain the B-mode power spectrum by running `ms_aps.ipynb`. The B-mode power spectrum is computed at
   ```
   clBB = local.compute_clbb(lbs,calC,Lmin=Lmin,Lmax=Lmax,lemin=lemin,lemax=lemax)
   ```

Reference
---
If you use this tool, please cite the following reference:

```
@article{Namikawa:2024dgj,
    author = "Namikawa, Toshiya",
    title = "{Exact CMB B-mode power spectrum from anisotropic cosmic birefringence}",
    eprint = "2404.13771",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "4",
    year = "2024"
}
```

Acknowledgement
---
This work is supported in part by JSPS KAKENHI Grant No. JP20H05859 and No. JP22K03682.
Part of this work uses resources of the National Energy Research Scientific Computing Center (NERSC).
The Kavli IPMU is supported by World Premier International Research Center Initiative (WPI Initiative), MEXT, Japan. 
