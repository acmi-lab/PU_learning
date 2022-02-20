import matplotlib.pyplot as plt 
import numpy as np
import sys 
from scipy.signal import savgol_filter
import argparse
import glob 

parser = argparse.ArgumentParser(description='Plot')
# parser.add_argument('--our', type=str, help='Name of the file')
# parser.add_argument('--nnPU', type=str, help='Name of the file')
# parser.add_argument('--uPU', type=str, help='Name of the file')
# parser.add_argument('--domain-discrimination', type=str, help='Name of the file')
parser.add_argument('--plot-name', type=str, help='Name of the file')
args = parser.parse_args()

seeds = [1432, 42, 8378]

net = "FCN"
log_dir = "logging_accuracy"
dataset = "cifar_binarized"

outfile_dd = glob.glob(f"{log_dir}/{dataset}/PvU_{net}*")
outfile_nnPU = glob.glob(f"{log_dir}/{dataset}/nnPU_{net}*")
outfile_uPU = glob.glob(f"{log_dir}/{dataset}/uPU_{net}*")
outfile_our = glob.glob(f"{log_dir}/{dataset}/TEDn_{net}*")
outfile_CVIR = glob.glob(f"{log_dir}/{dataset}/CVIR_{net}*")

test_acc_our = []
test_acc_CVIR = []
test_acc_nnPU = []
test_acc_uPU = []
test_acc_PvU = []
test_acc_dedpul = []

for i,seed in enumerate(seeds): 
    with open(outfile_our[i]) as f: 
        temp = []
        for line in f: 
            if (line.startswith("Warm_start")): 
                pass
            elif (line.startswith("Algo_training")):
                pass
            else: 
                vals = line.split(",")
                
                temp.append(float(vals[2]))
        test_acc_our.append(temp)

for i,seed in enumerate(seeds):
    with open(outfile_CVIR[i]) as f:
        temp = []
        for line in f:
            if (line.startswith("Warm_start")):
                pass
            elif (line.startswith("Algo_training")):
                pass
            else:
                vals = line.split(",")

                temp.append(float(vals[2]))
        test_acc_CVIR.append(temp)

for i,seed in enumerate(seeds): 
    with open(outfile_nnPU[i]) as f: 
        temp = []
        for line in f: 
            if (line.startswith("Warm_start")): 
                pass
            elif (line.startswith("Algo_training")):
                pass
            else: 
                vals = line.split(",")
                
                temp.append(float(vals[2]))

        test_acc_nnPU.append(temp)

for i,seed in enumerate(seeds):
    with open(outfile_uPU[i]) as f:
        temp = []
        for line in f:
            if (line.startswith("Warm_start")):
                pass
            elif (line.startswith("Algo_training")):
                pass
            else:
                vals = line.split(",")

                temp.append(float(vals[2]))

        test_acc_uPU.append(temp)


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

                temp1.append(float(vals[2]))
                temp2.append(float(vals[3]))
        test_acc_PvU.append(temp1)
        test_acc_dedpul.append(temp2)

l=3.0
fc=20

win_size=9
for i in range(len(seeds)):
    test_acc_our[i] = savgol_filter(test_acc_our[i][:1000], win_size, 1)
    test_acc_CVIR[i] = savgol_filter(test_acc_CVIR[i][:1000], win_size, 1)
    test_acc_dedpul[i] = savgol_filter(test_acc_dedpul[i][:1000], win_size, 1)
    test_acc_nnPU[i] = savgol_filter(test_acc_nnPU[i][:1000], win_size, 1)
    test_acc_uPU[i] = savgol_filter(test_acc_uPU[i][:1000], win_size, 1)
    test_acc_PvU[i] = savgol_filter(test_acc_PvU[i][:1000], win_size, 1)

test_acc_our_b = np.mean(test_acc_our, axis=0)
test_acc_our_var = np.std(test_acc_our, axis=0)

test_acc_CVIR_b = np.mean(test_acc_CVIR, axis=0)
test_acc_CVIR_var = np.std(test_acc_CVIR, axis=0)

test_acc_dedpul_b = np.mean(test_acc_dedpul, axis=0)
test_acc_dedpul_var = np.std(test_acc_dedpul, axis=0)

test_acc_nnPU_b = np.mean(test_acc_nnPU, axis=0)
test_acc_nnPU_var = np.std(test_acc_nnPU, axis=0)


test_acc_uPU_b = np.mean(test_acc_uPU, axis=0)
test_acc_uPU_var = np.std(test_acc_uPU, axis=0)


test_acc_PvU_b = np.mean(test_acc_PvU, axis=0)
test_acc_PvU_var = np.std(test_acc_PvU, axis=0)

fig,ax = plt.subplots()

#ax.plot(range(len(test_acc_CVIR_b)), test_acc_CVIR_b, linewidth=l, color='indigo', label="CVIR (" + r'$ \alpha^* $' + ")")
#ax.fill_between(range(len(test_acc_CVIR_b)), test_acc_CVIR_b - test_acc_CVIR_var, test_acc_CVIR_b + test_acc_CVIR_var, color='indigo', alpha = 0.3)


ax.plot(range(len(test_acc_our_b)), test_acc_our_b, linewidth=l, color='royalblue', label="(TED)$^n$")
ax.fill_between(range(len(test_acc_our_b)), test_acc_our_b - test_acc_our_var, test_acc_our_b + test_acc_our_var, color='royalblue', alpha = 0.3)


ax.plot(range(len(test_acc_dedpul_b)), test_acc_dedpul_b, linewidth=l, color='darkorange', label="Dedpul")
ax.fill_between(range(len(test_acc_dedpul_b)), test_acc_dedpul_b - test_acc_dedpul_var, test_acc_dedpul_b + test_acc_dedpul_var, color='darkorange', alpha = 0.3)

ax.plot(range(len(test_acc_nnPU_b)), test_acc_nnPU_b, linewidth=l, color='forestgreen', label="nnPU (" + r'$ \alpha^* $' + ")")
ax.fill_between(range(len(test_acc_our_b)), test_acc_nnPU_b - test_acc_nnPU_var, test_acc_nnPU_b + test_acc_nnPU_var, color='forestgreen', alpha = 0.3)

ax.plot(range(len(test_acc_uPU_b)), test_acc_uPU_b, linewidth=l, color='gray', label="uPU (" + r'$ \alpha^* $' + ")")
ax.fill_between(range(len(test_acc_uPU_b)), test_acc_uPU_b - test_acc_uPU_var, test_acc_uPU_b + test_acc_uPU_var, color='gray', alpha = 0.3)

ax.plot(range(len(test_acc_PvU_b)), test_acc_PvU_b, linewidth=l, color='crimson', label="PvU")
ax.fill_between(range(len(test_acc_PvU_b)), test_acc_PvU_b - test_acc_PvU_var, test_acc_PvU_b + test_acc_PvU_var, color='crimson', alpha = 0.3)


plt.xticks(np.arange(0, 1000+1, 250),fontsize=18)
plt.yticks(fontsize=18)

# plt.axvline(x=100, linestyle='--', linewidth=l)
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_xlabel('Epochs',fontsize=20)
plt.legend(prop={"size":18})
ax.legend()

plt.grid()
plt.savefig(args.plot_name ,transparent=True,bbox_inches='tight')
plt.clf()
