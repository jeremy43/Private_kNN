delta = 1e-8
sigma = 1
#import autodp
prob = 0.001
from autodp1.autodp import rdp_bank, dp_acct, rdp_acct, privacy_calibrator

gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
acct2 =rdp_acct.anaRDPacct()
for i in range(10000):
    acct2.compose_poisson_subsampled_mechanisms(gaussian, prob,coeff=1)
    if i % 200 == 0:
        print('i=',i, acct2.get_eps(delta))


acct1 = rdp_acct.anaRDPacct()
acct1.compose_poisson_subsampled_mechanisms(gaussian,prob,10000)
print('eps coefff=10000', acct1.get_eps(delta))
