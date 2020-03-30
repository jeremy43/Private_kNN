import numpy as np
import os
import scipy
import math
import sys

sys.path.append('../autodp/autodp')

import scipy.stats
import rdp_bank, dp_acct,rdp_acct,  privacy_calibrator

"""
Reproduce Figure 5
We compare the privacy cost of noisy screening by answering 8192 CIFAR10 queries with five randomized algorithms.
The sampling ratio = 0.25, noise_scale = 85, the number of teachers = 300 and the threshold = 210.


The five algorithms are:

Gaussian mechanism
Data-dependent screening
Data-independent screening
Poisson Subsampled Data-independent screening
Poisson Subsampled Gaussian 

For the data-independent screening, we find that the worst pair of neighboring datasets occur either around max{vote}=Threshold or around the boundaries when max{votes} = [k/c]. (See the discussion in Appendix B).
For efficiency, the RDP of data-independent screening used in this script (based on the above heuristic) is a fast approximation of the exact data-independent screening that we report in Theorem~7. Thus, the Figure 5 generate by this script is slightly different from Figure 5 in the main paper. 

"""




prob = 0.25
delta = 1e-5
filepath = 'knn_num_neighbor_300_figure_knn_voting.npy'
teachers_preds = np.load(filepath)
teachers_preds = teachers_preds
sigma = 85
threshold = 210

def figure_query_eps():
   


    eps_dependent = [] # Records the cost of Data-dependent screening
    dependent_acct = rdp_acct.anaRDPacct()

    eps_inde =[] # Records the cost of Data-independent screening
    inde_acct = rdp_acct.anaRDPacct()

    eps_subsample_gau = [] # Records the cost of Subsampled Gaussian
    subsample_acct = rdp_acct.anaRDPacct()

    eps_subsample_inde =[] # Records the cost of Subsampled Data-independent screening
    inde_subsample_acct = rdp_acct.anaRDPacct()

    eps_gau =[] # Records the cost of Gaussian mechanism
    gau_acct = rdp_acct.anaRDPacct()

    index = []


    last_compose = 0
    num_iteration = len(teachers_preds)
    klist = [2**i for i in range(int(math.floor(math.log(num_iteration,2)))+1)]
    gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    # func3 denote the data-independent RDP of noisy screening
    func3 = lambda x: rdp_bank.RDP_independent_noisy_screen({'k':300,'thresh':threshold,'sigma':sigma}, x)

    for i, vote in enumerate(teachers_preds):
        if i% 100 ==0:
            print('idx=',i)
        preds = vote
        preds = preds.astype(np.int64)
        label_count = np.bincount(preds, minlength=10)
        max_count = np.max(label_count)

        p = scipy.stats.norm.logsf(threshold - max_count, scale=sigma)
        q = scipy.stats.norm.logsf(threshold -1- max_count, scale=sigma)
        func = lambda x: rdp_bank.RDP_noisy_screen({'logp':p,'logq':q}, x)
        dependent_acct.compose_mechanism(func, coeff=1)
        gau_acct.compose_mechanism(gaussian)
        subsample_acct.compose_poisson_subsampled_mechanisms1(gaussian, prob, coeff=1)
        if i + 1 in klist:
            inde_acct.compose_mechanism(func3, coeff= i+1 - last_compose)
            inde_subsample_acct.compose_poisson_subsampled_mechanisms1(func3, prob, coeff=i +1 -last_compose)
            eps_gau.append(gau_acct.get_eps(delta))
            eps_subsample_gau.append(subsample_acct.get_eps(delta))
            eps_dependent.append(dependent_acct.get_eps(delta))
            eps_inde.append(inde_acct.get_eps(delta))
            eps_subsample_inde.append(inde_subsample_acct.get_eps(delta))
            index.append(i)
            last_compose = i + 1
            print('idx =', i)
    # Save the records
    log = {}
    log['eps_gau'] = eps_gau
    log['eps_dependent'] = eps_dependent
    log['eps_inde'] = eps_inde
    log['eps_subsample_inde'] = eps_subsample_inde
    log['eps_subsample_gau'] = eps_subsample_gau
    log['index'] = klist
    file_path = 'new_independent.pkl'
    import pickle
    with open(file_path,'wb') as f:
        pickle.dump(log,f)
figure_query_eps()

def draw_figure():


    file_path = 'new_independent.pkl'
    import pickle
    with open(file_path,'rb') as f:
        log = pickle.load(f)

    eps_gau = log['eps_gau']

    eps_dependent = log['eps_dependent']
    eps_inde =log['eps_inde']
    eps_subsample_gau =log['eps_subsample_gau']
    eps_subsample_inde = log['eps_subsample_inde']
    klist = log['index']
    print('klist',klist)
    print('dependent',eps_dependent)
    print('gau',eps_subsample_gau)
    print('naive gaussian',eps_gau)
    print('compose independent  without subsample',eps_inde)
    print('subsample independent ',eps_subsample_inde)

    import matplotlib
    import matplotlib.pyplot as plt

    font = {'family': 'times',
            'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)
    plt.rc('axes', labelsize=27)
    plt.rc('axes', titlesize=20)
    plt.rc('xtick', labelsize=27)
    plt.rc('ytick', labelsize=27)

    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    plt.plot(klist, eps_gau, '-g^',linewidth=2)
    plt.plot(klist, eps_subsample_inde, '-ro', linewidth=2)
    plt.plot(klist,eps_dependent,'--b',linewidth=2)
    plt.plot(klist,eps_inde ,'-b',linewidth=2)
    plt.plot(klist,eps_subsample_gau,'--k',linewidth=2)


    #plt.title('Comparison of Different Randomized Algorithms in Noisy Screening with Sampling Ratio $\gamma = 0.25$ ')
    plt.grid(True)
    plt.xlabel(r'Number of queries for screening')
    plt.ylabel(r'Privacy cost $\epsilon$')
    plt.legend([r'Gaussian Mechanism','Poisson Subsampled Data-independent Screening','Data-dependent Screening','Data-independent Screening','Poisson Subsampled Gaussian'], loc='best')
    plt.show()
    plt.savefig('tight_iteration2eps.png',bbox_inches='tight')

draw_figure()
