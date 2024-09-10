# Code associated with Schraiber et al (2024) "Unifying approaches from statistical genetics and phylogenetics formapping phenotypes in structured populations"

The code here can be used to reproduce the figures found in the above manuscript.

## Figure 2B-C
Code is found in cells 5--9 of `phylogeny_simulations.ipynb`, which contain code to both perform Brownian motion simulations and inferences, as well as generate the figures.

## Figure 2D-E
Data can be generated with `run_reml.py`, which both simulates data and runs a custom REML implementation to infer variance components and fixed effects. 
Data are not included in the repository due to the filesize.
Figures can plotted using cells 15-30 in `gwas_figures.ipynb`.

## Figure 3
Data can be generated with `run_reml_tree.py`, which both simulates data and runs a custom REML implementation to infer variance components and fixed effects. 
Data are not included in the repository due to the filesize.
Figures can plotted using cells 23-31 in `gwas_figures.ipynb`.

## Figure 4
Data can be found 
