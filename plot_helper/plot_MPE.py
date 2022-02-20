import matplotlib.pyplot as plt 
import numpy as np
import sys 
from scipy.signal import savgol_filter
import argparse
import glob 

parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('--plot-name', type=str, help='Name of the file')
args = parser.parse_args()


seeds = [1432, 42, 8378]
alpha = 0.5 
outfile_our= [] 
# outfile_uPU= []
# outfile_nnPU= []
outfile_dd= []


net = "FCN"
log_dir = "logging_accuracy"
dataset = "cifar_binarized"

outfile_dd = glob.glob(f"{log_dir}/{dataset}/PvU_{net}*")
outfile_our = glob.glob(f"{log_dir}/{dataset}/TEDn_{net}*")


mpe_estimate_our = []
mpe_estimate_dedpul = []
mpe_estimate_en = []

for i,seed in enumerate(seeds): 
    with open(outfile_our[i]) as f: 
        temp = []
        idx = 4
        for line in f: 
            if (line.startswith("Warm_start")): 
                pass
            elif (line.startswith("Algo_training")):
                idx = 3
                pass
            else: 
                vals = line.split(",")

                temp.append(float(vals[idx]))
        mpe_estimate_our.append(temp)

for i,seed in enumerate(seeds): 
    with open(outfile_dd[i]) as f: 
        temp1 = []
        temp2 = []

        for line in f: 
            if (line.startswith("Warm_start")): 
                pass
            elif (line.startswith("Algo_training")):
                pass
            else: 
                vals = line.split(",")

                # if args.one_minus:
                    # mpe_estimate_dedpul.append(1.0 - float(vals[5]))
                # else: 
                temp1.append(float(vals[5]))
                temp2.append(float(vals[6]))
        mpe_estimate_dedpul.append(temp1)
        mpe_estimate_en.append(temp2)


import scipy.io as sio

alphamax = sio.loadmat('Alphamax.mat')['alphas'][0]

l=3.0
fc=20

win_size =9
for i in range(len(seeds)):
    mpe_estimate_our[i] =  savgol_filter(mpe_estimate_our[i][0:1000], win_size, 1)
    mpe_estimate_dedpul[i] =  savgol_filter(mpe_estimate_dedpul[i][:1000], win_size, 1)
    mpe_estimate_en[i]=  savgol_filter(mpe_estimate_en[i][:1000], win_size, 1)
    mpe_estimate_alphamax=  savgol_filter(alphamax, 3, 1)

fig,ax = plt.subplots()

ax.set_xlabel('Epochs',fontsize=20)
ax.set_ylabel('Mixture Proportion',fontsize=20)

mpe_estimate_our_b = np.mean(mpe_estimate_our, axis=0)
mpe_estimate_our_var = np.std(mpe_estimate_our, axis=0)

mpe_estimate_dedpul_b = np.mean(mpe_estimate_dedpul, axis=0)
mpe_estimate_dedpul_var = np.std(mpe_estimate_dedpul, axis=0)

mpe_estimate_en_b = np.mean(mpe_estimate_en, axis=0)
mpe_estimate_en_var = np.std(mpe_estimate_en, axis=0)

mpe_estimate_alphamax_b = mpe_estimate_alphamax
# test_acc_EN_var = np.var(test_acc_EN, axis=0)

ax.plot(range(len(mpe_estimate_our_b)), mpe_estimate_our_b, linewidth=l, color='royalblue', label="(TED)$^n$")
ax.fill_between(range(len(mpe_estimate_our_b)), mpe_estimate_our_b - mpe_estimate_our_var, mpe_estimate_our_b + mpe_estimate_our_var, color='royalblue', alpha = 0.3)

ax.plot(range(len(mpe_estimate_dedpul_b)), mpe_estimate_dedpul_b, linewidth=l, color='darkorange', label="DEDPUL")
ax.fill_between(range(len(mpe_estimate_dedpul_b)), mpe_estimate_dedpul_b - mpe_estimate_dedpul_var, mpe_estimate_dedpul_b + mpe_estimate_dedpul_var, color='darkorange', alpha = 0.3)


ax.plot(range(len(mpe_estimate_en_b)), mpe_estimate_en_b, linewidth=l, color='crimson', label="EN")
ax.fill_between(range(len(mpe_estimate_en_b)), mpe_estimate_en_b - mpe_estimate_en_var, mpe_estimate_en_b + mpe_estimate_en_var, color='crimson', alpha = 0.3)

#ax.plot(range(0, 2000, 5 ), mpe_estimate_alphamax_b, linewidth=l, color='forestgreen', label="Alphamax (" + r'$ \alpha^* $' + ")")
# ax.fill_between(range(len(test_acc_our_b)), test_acc_nnPU_b - test_acc_nnPU_var, test_acc_nnPU_b + test_acc_nnPU_var, color='forestgreen', alpha = 0.3)


#ax.plot(range(0,2000, 5), mpe_estimate_alphamax, linewidth=l, label = "Alphamax", color='green')
ax.axhline(y=alpha, linestyle='--', linewidth=l, label="True MPE", color='black')
ax.set_ylim((0.0, 1.0))

plt.xticks(np.arange(0, 1000+1, 250),fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop={"size":18})

plt.grid()
ax.legend()
plt.savefig(args.plot_name ,transparent=True,bbox_inches='tight')
plt.clf()
