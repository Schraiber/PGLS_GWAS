import msprime as msp
import numpy as np
import pandas as pd
import argparse
import scipy.stats as st
import scipy.optimize as opt

parser = argparse.ArgumentParser("Simulate phenotypes from a two population split neutral model and output for GCTA")
parser.add_argument("-n", default = 10000, type = float, help = "Diploid population size (default = 10000)")
parser.add_argument("-r", default = 1e-5, type = float, help = "Recombination rate (default = 1e-5)")
parser.add_argument("-m", default = 1e-6, type = float, help = "Mutation rate (default = 1e-6)")
parser.add_argument("-l", default = 100000, type = float, help = "Sequence length (default = 100000)")
parser.add_argument("-a", default = 100, type = float, help = "Diploid sample size in population 1 (default = 50)")
parser.add_argument("-b", default = 100, type = float, help = "Diploid sample size in population 2 (default = 50)")
parser.add_argument("-s", default = 0.01, type = float,  help = "Standard deviation of effect size distribution (default = 0.01)")
parser.add_argument("-v", default = 3, type = float, help = "Environmental variance (default = 3)")
parser.add_argument("-e", default = 0, type = float,help = "Standard deviation of environmental shift in population 1 (default = 0)")
parser.add_argument("-t", required = True, type = float,  help = "Split time (in generations)")
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
demography = msp.Demography()
demography.add_population(name="A",initial_size=args.n)
demography.add_population(name="B",initial_size=args.n)
demography.add_population(name="C",initial_size=args.n)
demography.add_population_split(time=args.t,derived=["A","B"],ancestral="C")

#generate SNPs for phenotype

print("Generate SNPs for phenotype")

#Get trees
ts = msp.sim_ancestry({"A":args.a,"B":args.b}, demography=demography,recombination_rate=args.r,sequence_length=args.l)

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
    pheno += np.random.normal(0,np.sqrt(args.v),args.a+args.b) #NB: the environmental variance parameter is the VARIANCE, so need square root. 

#Offset phenos if necessary
if args.e > 0:
    env_mean = np.array([np.random.normal(0,args.e,1)[0]]*args.a + [0]*args.b) #the factor 2 is due to the stupid diploidy thing
    pheno += env_mean

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
ts = msp.sim_ancestry({"A":args.a,"B":args.b}, demography=demography,recombination_rate=1e-5,sequence_length=1000)
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

print("Generating design matrices")

#generate design matrix with just SNP
x_SNP = np.vstack((np.array([1]*len(pheno)),test_SNP)).T

#generate design matrix with SNP + PC1
x_SNP_PC = np.vstack((np.array([1]*len(pheno)),test_SNP,eigenvectors[:,0])).T

#get SNP, OLS
print("Running SNP, OLS")
fix_OLS, var_OLS = optimize_LL_reml(x_SNP,GRM_norm,pheno,useGRM=False,verbose=True)

#get SNP+PC, OLS
print("Running SNP+PC, OLS")
fix_OLS_PC, var_OLS_PC = optimize_LL_reml(x_SNP_PC,GRM_norm,pheno,useGRM=False,verbose=True)

#get SNP, GLS
print("Running SNP, GLS")
fix_GLS, var_GLS = optimize_LL_reml(x_SNP,GRM_norm,pheno,useGRM=True,verbose=True)

#get SNP+PC, GLS
print("RUnning SNP+PC, GLS")
fix_GLS_PC, var_GLS_PC = optimize_LL_reml(x_SNP_PC,GRM_norm,pheno,useGRM=True,verbose=True)

#make output
print("Prepping output")
fix = np.vstack((fix_OLS,fix_OLS_PC,fix_GLS,fix_GLS_PC))
res_fix = pd.DataFrame(
    {"method":["OLS","OLS","OLS_PC","OLS_PC","OLS_PC","GLS","GLS","GLS_PC","GLS_PC","GLS_PC"],
     "term":["Intercept","X","Intercept","X","PC1","Intercept","X","Intercept","X","PC1"],
     "estimate":fix[:,0],
     "se":fix[:,1]
    }
)

var = np.vstack((var_OLS,var_OLS_PC,var_GLS,var_GLS_PC))
res_var = pd.DataFrame(
       {"method":["OLS","OLS_PC","GLS","GLS_PC"],"sigma2g":var[:,0],"sigma2e":var[:,1],"LL":var[:,2]}
)

#write output
print("Writing output")
res_fix.to_csv(args.o+".fixeff.txt",sep="\t",index=False)
res_var.to_csv(args.o+".var.txt",sep="\t",index=False)
