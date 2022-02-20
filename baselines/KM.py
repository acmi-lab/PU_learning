"""
DO MIXTURE PROPORTION ESTIMATION 
Using gradient thresholding of the $\C_S$-distance
"""
from cvxopt import matrix, solvers, spmatrix
from math import sqrt
import numpy as np
import scipy.linalg as scilin
from sklearn.decomposition import PCA


def find_nearest_valid_distribution(u_alpha, kernel, initial=None, reg=0):
    """ (solution,distance_sqd)=find_nearest_valid_distribution(u_alpha,kernel):
    Given a n-vector u_alpha summing to 1, with negative terms, 
    finds the distance (squared) to the nearest n-vector summing to 1, 
    with non-neg terms. Distance calculated using nxn matrix kernel. 
    Regularization parameter reg -- 

    min_v (u_alpha - v)^\top K (u_alpha - v) + reg* v^\top v"""

    P = matrix(2 * kernel)
    n = kernel.shape[0]
    q = matrix(np.dot(-2 * kernel, u_alpha))
    A = matrix(np.ones((1, n)))
    b = matrix(1.)
    G = spmatrix(-1., range(n), range(n))
    h = matrix(np.zeros(n))
    dims = {'l': n, 'q': [], 's': []}
    solvers.options['show_progress'] = False
    solution = solvers.coneqp(
        P,
        q,
        G,
        h,
        dims,
        A,
        b,
        initvals=initial
        )
    distance_sqd = solution['primal objective'] + np.dot(u_alpha.T,
            np.dot(kernel, u_alpha))[0, 0]
    return (solution, distance_sqd)


def get_distance_curve(
    kernel,
    lambda_values,
    N,
    M=None,
    ):
    """ Given number of elements per class, full kernel (with first N rows corr.
    to mixture and the last M rows corr. to component, and set of lambda values
    compute $\hat d(\lambda)$ for those values of lambda"""

    d_lambda = []
    if M == None:
        M = kernel.shape[0] - N
    prev_soln=None    
    for lambda_value in lambda_values:
        u_lambda = lambda_value / N * np.concatenate((np.ones((N, 1)),
                np.zeros((M, 1)))) + (1 - lambda_value) / M \
            * np.concatenate((np.zeros((N, 1)), np.ones((M, 1))))
        (solution, distance_sqd) = \
            find_nearest_valid_distribution(u_lambda, kernel, initial=prev_soln)
        prev_soln = solution
        if distance_sqd < 0.0:
            print(distance_sqd)
        d_lambda.append(sqrt(distance_sqd))
    d_lambda = np.array(d_lambda)
    return d_lambda





def compute_best_rbf_kernel_width(X_mixture,X_component):        
    N=X_mixture.shape[0]
    M=X_component.shape[0]
    # compute median of pairwise distances
    X=np.concatenate((X_mixture,X_component))
    dot_prod_matrix=np.dot(X,X.T)
    norms_squared=sum(np.multiply(X,X).T)    
    distance_sqd_matrix=np.tile(norms_squared,(N+M,1)) + \
        np.tile(norms_squared,(N+M,1)).T - 2*dot_prod_matrix            
    kernel_width_median = sqrt(np.median(distance_sqd_matrix))
    kernel_width_vals= np.logspace(-1,1,5) * kernel_width_median
    
    # Find best kernel width
        
    max_dist_RKHS=0
    for kernel_width in kernel_width_vals: 
        kernel=np.exp(-distance_sqd_matrix/(2.*kernel_width**2.))        
        dist_diff = np.concatenate((np.ones((N, 1)) / N, 
                                    -1 * np.ones((M,1)) / M))
        temp = np.dot(dist_diff.T, np.dot(kernel, dist_diff))[0,0]
        if temp <0.0: 
            print(temp)
            continue
        distribution_RKHS_distance = sqrt(temp)
        if distribution_RKHS_distance > max_dist_RKHS:
            max_dist_RKHS=distribution_RKHS_distance
            best_kernel_width=kernel_width                
    kernel=np.exp(-distance_sqd_matrix/(2.*best_kernel_width**2.))
    return best_kernel_width,kernel

        
def mpe(kernel,N,M,nu,epsilon=0.04,lambda_upper_bound=8.):
    """ Do mixture proportion estimation (as in paper)for N  points from  
    mixture F and M points from component H, given kernel of size (N+M)x(N+M), 
    with first N points from  the mixture  and last M points from 
    the component, and return estimate of lambda_star where
    G =lambda_star*F + (1-lambda_star)*H"""

    dist_diff = np.concatenate((np.ones((N, 1)) / N, -1 * np.ones((M,1)) / M))
    distribution_RKHS_distance = sqrt(np.dot(dist_diff.T, 
                                    np.dot(kernel, dist_diff))[0,0])
    lambda_left=1.
    lambda_right=lambda_upper_bound        
    while lambda_right-lambda_left>epsilon:
        lambda_value=(lambda_left+lambda_right)/2.        
        u_lambda = lambda_value / N * np.concatenate((np.ones((N, 1)),
            np.zeros((M, 1)))) + (1 - lambda_value) / M \
            * np.concatenate((np.zeros((N, 1)), np.ones((M, 1))))
        (solution, distance_sqd) = \
              find_nearest_valid_distribution(u_lambda, kernel)
        d_lambda_1=sqrt(distance_sqd)
        
        lambda_value=(lambda_left+lambda_right)/2. + epsilon/2.        
        u_lambda = lambda_value / N * np.concatenate((np.ones((N, 1)),
            np.zeros((M, 1)))) + (1 - lambda_value) / M \
            * np.concatenate((np.zeros((N, 1)), np.ones((M, 1))))
        (solution, distance_sqd) = \
             find_nearest_valid_distribution(u_lambda, kernel)
        d_lambda_2=sqrt(distance_sqd)
            
        slope_lambda=(d_lambda_2 - d_lambda_1)*2./epsilon                    
    
        if slope_lambda > nu*distribution_RKHS_distance:
            lambda_right=(lambda_left+lambda_right)/2.
        else:
            lambda_left=(lambda_left+lambda_right)/2.
            
    return (lambda_left+lambda_right)/2.
                                        

def wrapper(X_mixture,X_component):                 
    """ Takes in 2 arrays containing the mixture and component data as 
    numpy arrays, and prints the estimate of kappastars using the two gradient 
    thresholds as detailed in the paper as KM1 and KM2"""
    
    N=X_mixture.shape[0]                                                                     
    M=X_component.shape[0]
    best_width,kernel=compute_best_rbf_kernel_width(X_mixture,X_component)                 
    lambda_values=np.array([1.00,1.05])                                
    dists=get_distance_curve(kernel,lambda_values,N=N,M=M)
    begin_slope=(dists[1]-dists[0])/(lambda_values[1]-lambda_values[0])
    dist_diff = np.concatenate((np.ones((N, 1)) / N, -1 * np.ones((M,1)) / M))
    distribution_RKHS_dist = sqrt(np.dot(dist_diff.T, np.dot(kernel, dist_diff))[0,0])
    thres_par=0.2    
    nu1=(1-thres_par)*begin_slope + thres_par*distribution_RKHS_dist
    nu1=nu1/distribution_RKHS_dist                                    
    lambda_star_est_1=mpe(kernel,N,M,nu=nu1)
    kappa_star_est_1=(lambda_star_est_1-1)/lambda_star_est_1                    
    nu2=1/sqrt(np.min([M,N]))
    nu2=nu2/distribution_RKHS_dist
    if nu2>0.9:
        nu2=nu1
    lambda_star_est_2=mpe(kernel,N,M,nu=nu2)                    
    kappa_star_est_2=(lambda_star_est_2-1)/lambda_star_est_2
    return (kappa_star_est_2,kappa_star_est_1)	

def KM_estimate(p_data, u_data, data_type):
    if not data_type.startswith("UCI"):
        pca = PCA(n_components=50, svd_solver='full')
        new_X = pca.fit_transform(np.concatenate((u_data, p_data), axis=0))

        X_mixture = new_X[0:len(u_data)]
        X_component = new_X[len(u_data):]
    else: 
        X_mixture = u_data 
        X_component = p_data
    

    n_shuffle = np.random.permutation(len(X_mixture))
    p_shuffle = np.random.permutation(len(X_component))

    X_component = X_component[p_shuffle][:2000]
    X_mixture = X_mixture[n_shuffle][:2000]

    return wrapper(X_mixture, X_component)
