import msprime as msp
import numpy as np
import pandas as pd
import argparse
import scipy.stats as st
import scipy.optimize as opt

parser = argparse.ArgumentParser("Simulate phenotypes from a species tree")
parser.add_argument("-t", required = True, help = "Newick file with species tree")
parser.add_argument("-n", default = 10000, type = float, help = "Diploid population size (default = 10000)")
parser.add_argument("-r", default = 1e-5, type = float, help = "Recombination rate (default = 1e-5)")
parser.add_argument("-m", default = 1e-6, type = float, help = "Mutation rate (default = 1e-6)")
parser.add_argument("-l", default = 100000, type = float, help = "Sequence length (default = 100000)")
parser.add_argument("-a", default = 10, type = float, help = "Sample size in each population")
parser.add_argument("-s", default = 0.01, type = float,  help = "Standard deviation of effect size distribution (default = 0.01)")
parser.add_argument("-v", default = 10, type = float, help = "Environmental variance (default = 10)")
parser.add_argument("-o", required = True, help = "Output prefix")

args = parser.parse_args()

#define functions
def my_mvn_ll(x,Sigma):
    SigmaInv = np.linalg.inv(Sigma)
    return -1/2*(np.linalg.slogdet(Sigma)[1]+x.T.dot(SigmaInv).dot(x)+len(x)*np.log(2*np.pi))

def LL_reml(A, sigma2g, sigma2e, GRM, c, verbose = False):
    Sigma = sigma2g*GRM + sigma2e*np.identity(GRM.shape[0])
    
    SigmaNew = A.dot(Sigma).dot(A.T) #Note that this is inverted relative to the notes at https://dnett.github.io/S510/20REML.pdf because I define A to have ROWS (c.f. slide 8)

    #LL = -st.multivariate_normal(None,SigmaNew,allow_singular=True).logpdf(c)

    LL = -my_mvn_ll(c,SigmaNew)
    
    if verbose: print(sigma2g, sigma2e, LL)
    
    return LL

def optimize_LL_reml(x,GRM,pheno,useGRM=True,verbose=False):
    #first, compute the reml matrices, these only need to be computed once
    print("Computing matrices")
    #Convert x to np array
    x = np.array(x)

    #compute the projection matrix
    xtxinv = np.linalg.inv(x.transpose().dot(x))
    Px = x.dot(xtxinv).dot(x.transpose())

    #Compute the A matrix
    A = (np.identity(len(pheno))-Px)
    #This part finds only the linearly independent rows of A (see https://stackoverflow.com/questions/53667174/elimination-the-linear-dependent-columns-of-a-non-square-matrix-in-python)
    Q,R = np.linalg.qr(A.T)
    A = A[np.abs(np.diag(R))>=1e-10]

    #compute the error contrasts
    c = A.dot(pheno)

    print("Running optimization")

    if useGRM:
        optimization = opt.fmin_l_bfgs_b(lambda x: LL_reml(A,x[0],x[1],GRM,c,verbose=verbose),np.array([10,10]),bounds=np.array([[1e-5,100],[1e-5,100]]),approx_grad=True)

        #get parameters
        sigma2g = optimization[0][0]
        sigma2e = optimization[0][1]
        LL = optimization[1]
    else:
        optimization = opt.fmin_l_bfgs_b(lambda x: LL_reml(A,0,x[0],GRM,c,verbose=verbose),np.array([10]),bounds=np.array([[1e-5,100]]),approx_grad=True)
        sigma2g = 0
        sigma2e = optimization[0][0]
        LL = optimization[1]

    print("Getting fixed effects")

    Sigma = sigma2g*GRM + sigma2e*np.identity(len(pheno))

    SigmaInv = np.linalg.inv(Sigma)

    xtSigmaxinv = np.linalg.inv(x.T.dot(SigmaInv).dot(x))

    betaHat = xtSigmaxinv.dot(x.T).dot(SigmaInv).dot(pheno)

    covBetaHat = xtSigmaxinv

    seBetaHat = np.real(np.sqrt(np.diagonal(covBetaHat))) #NB: sometimes returns a complex value with imaginary component = 0

    fixeff = np.vstack((betaHat,seBetaHat)).T #column 1 is the point estimate, column 2 is the se

    varComponents = np.array([sigma2g,sigma2e,-LL])

    return fixeff, varComponents

print("Running")

print("Will output to prefix " + args.o)

#make demography
tree_newick = str(np.loadtxt(args.t,dtype=str))
demography = msp.Demography.from_species_tree(tree_newick, args.n)
num_pop = (demography.num_populations+1)//2
samples = {("t"+str(i)):args.a for i in range(1,num_pop+1)}

#generate SNPs for phenotype

print("Generate SNPs for phenotype")

#Get trees
ts = msp.sim_ancestry(samples,demography=demography,recombination_rate=1e-5,sequence_length=100000)

#Get mutations
muts = msp.sim_mutations(ts,rate=args.m,discrete_genome=False) #NB: Discrete_genome = False is essential for infinite sites mutation!

#Get mutation matrix
mut_mat = muts.genotype_matrix().transpose()

#make diploid mutation matrix
even_rows = np.arange(mut_mat.shape[0]/2,dtype=int)*2
odd_rows = even_rows+1
mut_mat = mut_mat[even_rows,:]+mut_mat[odd_rows,:]

#Get effect sizes under GCTA model
AF = np.mean(mut_mat,axis=0)/2
beta = np.random.normal(0,args.s/np.sqrt(2*AF*(1-AF)),size=mut_mat.shape[1]) #nb: second argument is standard deviation

#Get phenos

print("Make phenotypes")

pheno = mut_mat.dot(beta)
if args.v>0:
    pheno += np.random.normal(0,np.sqrt(args.v),len(pheno)) #NB: the environmental variance parameter is the VARIANCE, so need square root. 

#Get GRM
print("Getting GRM")
mut_mat_standardize = (mut_mat/np.sqrt(2*AF*(1-AF))) #Standardize the mutation matrix (maybe needs to be 2*AF*(1-AF)?
norm_mut_mat_standardize = mut_mat_standardize - np.mean(mut_mat_standardize,axis=0)
GRM_standardize = norm_mut_mat_standardize.dot(norm_mut_mat_standardize.transpose())
GRM_norm = GRM_standardize/mut_mat.shape[1]

#Get PCs
print("Getting PCs")
eigenvalues, eigenvectors = np.linalg.eig(GRM_norm)

#Generate null SNPs, of which we will select one for testing
#Note that the parameters here don't really matter, we just need to draw one SNP
print("Getting null SNP")
ts = msp.sim_ancestry(samples,demography=demography,recombination_rate=1e-5,sequence_length=1000)
muts = msp.sim_mutations(ts,rate=1e-6,discrete_genome=False) #NB: discrete_genome!
mut_mat = muts.genotype_matrix().transpose()
#make diploids
even_rows = np.arange(mut_mat.shape[0]/2,dtype=int)*2
odd_rows = even_rows+1
mut_mat = mut_mat[even_rows,:]+mut_mat[odd_rows,:]


#Get MAFs so we make sure to choose one with a MAF over 10%
AF = np.mean(mut_mat,axis=0)
AF = np.array([f if f<.5 else 1-f for f in AF])

#Pick a SNP
test_SNP = mut_mat[:,AF>.1][:,1]

print("Generating design matrix with SNP")

#generate design matrix with just SNP
x_SNP = np.vstack((np.array([1]*len(pheno)),test_SNP)).T

fix_noGRM = []
fix_GRM = []

print("Running OLS with just SNP")
#run OLS with just SNP
fix_standardize,var_standardize = optimize_LL_reml(x_SNP,GRM_norm,pheno,verbose=True,useGRM=False)
fix_noGRM.append(fix_standardize[1,:])

print("Running GLS with just SNP")
#run GLS with just SNP
fix_standardize,var_standardize = optimize_LL_reml(x_SNP,GRM_norm,pheno,verbose=True,useGRM=True)
fix_GRM.append(fix_standardize[1,:])

print("Running PCs")
for i in range(1,51):
    print("Running up to PC " + str(i))
    x_SNP_PC = np.vstack((np.array([1]*len(pheno)),test_SNP,eigenvectors[:,:i].T)).T
   
    print("Running OLS") 
    fix_standardize,var_standardize = optimize_LL_reml(x_SNP_PC,GRM_norm,pheno,verbose=True,useGRM=False)
    fix_noGRM.append(fix_standardize[1,:])
    
    print("Running GLS")
    fix_standardize,var_standardize = optimize_LL_reml(x_SNP_PC,GRM_norm,pheno,verbose=True,useGRM=True)
    fix_GRM.append(fix_standardize[1,:])
    
    print(np.array(fix_noGRM))

#make output
print("Prepping output")
fix_noGRM = np.array(fix_noGRM)
res_noGRM = pd.DataFrame({"estimate":fix_noGRM[:,0],"se":fix_noGRM[:,1],"PC":range(fix_noGRM.shape[0]),"method":"OLS"})

fix_GRM = np.array(fix_GRM)
res_GRM = pd.DataFrame({"estimate":fix_GRM[:,0],"se":fix_GRM[:,1],"PC":range(fix_GRM.shape[0]),"method":"GLS"})

res = pd.concat((res_noGRM,res_GRM),ignore_index=True)

#write output
print("Writing output")
res.to_csv(args.o+".fixeff.txt",sep="\t",index=False)
