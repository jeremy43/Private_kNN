delta = 1e-5
sigma = 85
import autodp
prob = 0.3
import numpy as np
import math
import cifar10_config as config
config = config.config
from autodp1.autodp import rdp_bank, dp_acct, rdp_acct, privacy_calibrator
filepath = 'save_model/knn_num_neighbor_300_figure_knn_voting.npy'
#filepath = 'save_model/knn_num_neighbor_300_knn_voting.npy'
teachers_preds = np.load(filepath)
teachers_preds = teachers_preds

gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
acct2 =rdp_acct.anaRDPacct()
acct3 = rdp_acct.anaRDPacct()
ber_acct =  rdp_acct.anaRDPacct()
ber_inde_acct = rdp_acct.anaRDPacct()
naive_gau_acct = rdp_acct.anaRDPacct()
num_iteration = len(teachers_preds)
klist = [2 ** i for i in range(int(math.floor(math.log(num_iteration, 2))) + 1)]
gau_eps_list =[]
ber_eps_list = []
ber_inde_eps_list = []
naive_gau_list =[]

for i, vote in enumerate(teachers_preds):
    if i % 100 == 0:
        print('idx=', i)
    preds = vote
    preds = preds.astype(np.int64)
    label_count = np.bincount(preds, minlength=10)
    max_count = np.max(label_count)
    import scipy.stats
    acct2.compose_poisson_subsampled_mechanisms(gaussian, prob,coeff=1)
    p = scipy.stats.norm.logsf(config.threshold - max_count, scale=sigma)
    q = scipy.stats.norm.logsf(config.threshold -1- max_count, scale=sigma)
    func = lambda x: rdp_bank.svt_gaussian({'logp':p,'logq':q}, x)
    #ber_acct.compose_poisson_subsampled_mechanisms(func, prob,coeff=1)
    p = scipy.stats.norm.logsf(0, scale=sigma)
    q = scipy.stats.norm.logsf(1, scale=sigma)
    func = lambda x: rdp_bank.svt_gaussian({'logp': p, 'logq': q}, x)
    ber_inde_acct.compose_poisson_subsampled_mechanisms(func, prob, coeff=1)
    naive_gau_acct.compose_mechanism(gaussian)


    if i + 1 in klist:
        gau_eps = acct2.get_eps(delta)
        ber_eps = ber_acct.get_eps(delta)
        ber_inde_eps = ber_inde_acct.get_eps(delta)
        naive_gau_eps = naive_gau_acct.get_eps(delta)
        print('i=',i,'gaussian', gau_eps)
        print('i=',i,'ber',ber_eps)
        print('i=',i,'indenpendent',ber_inde_eps)
        gau_eps_list.append(gau_eps)
        ber_eps_list.append(ber_eps)
        ber_inde_eps_list.append(ber_inde_eps)
        naive_gau_list.append(naive_gau_eps)




log = {}
log['eps_gau'] = gau_eps_list
log['eps_bern_dependent'] = ber_eps_list
log['eps_bern_inde'] = ber_inde_eps_list
log['eps_naive'] = naive_gau_list

log['index'] = klist
file_path = 'new_10000_figure_iter2eps.pkl'
import pickle
with open(file_path,'wb') as f:
    pickle.dump(log,f)

acct1 = rdp_acct.anaRDPacct()
for i in range(1001):
    acct2.compose_poisson_subsampled_mechanisms(gaussian,prob,coeff = 1)

    if i %100 == 0:
        print('i=',i,acct2.get_eps(delta))
    
acct1.compose_poisson_subsampled_mechanisms(gaussian,prob,coeff =1001)
#alpha = [i for i in range(2, 100)]
#alpha = np.array(alpha, dtype=np.int32)
#print(alpha)
#lalpha = 5

#print(acct1.get_rdp(alpha))
#print(acct2.get_rdp(alpha))
#print('forloop rdpint ',acct2.RDPs_int)
#print('rdpint acct1',acct1.RDPs_int)
#print('eps coefff=500', acct1.get_eps(delta))
#print('general psson', acct3.get_eps(delta))
