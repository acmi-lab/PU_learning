import numpy as np
import argparse
from scipy.signal import savgol_filter
import glob

parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('--alpha' ,type=float, default=0.5, help='True alpha')
args = parser.parse_args()

def return_arr(outfile):
    arr = []
    ignore = False
    with open(outfile, 'r') as f: 
        for line in f:
                if (line.startswith("Warm_start")):
                    ignore=True
                    pass
                elif (line.startswith("Algo_training")):
                    ignore=False
                    pass
                else:
                    if not ignore: 
                        vals = line.split(",")

                        temp_arr = []
                        for val in vals:
                            temp_arr.append(float(val))

                        arr.append(temp_arr)
    arr = np.array(arr)
    print(arr.shape)
    return arr[:1000] 

def get_method_contents(outfile_filename):
    return_arrs = []

    for filename in outfile_filename: 
        print(filename)
        return_arrs.append(return_arr(filename))

    return_arrs = np.array(return_arrs)
    return return_arrs

def return_acc(arr): 
    acc = savgol_filter(arr,9, 1, axis=1)
    return acc

def return_mpe(arr):
    mpe = savgol_filter(np.abs(arr - args.alpha), 9, 1, axis=1)
    return mpe 

def oracle_results(acc, mpe,orig_acc):
    idx = np.argmax(orig_acc[:, :, 2], axis=1)
    print(idx)

    max_acc = []
    mpe_at_max_acc = [] 
    for i in range(len(idx)): 
        max_acc.append(acc[i,idx[i]])
        mpe_at_max_acc.append(mpe[i, idx[i]])

    max_acc = np.array(max_acc)
    mpe_at_max_acc = np.array(mpe_at_max_acc)

    return max_acc, mpe_at_max_acc

def converged_results(acc, mpe): 
    idx = -10
    return acc[:, idx, :], mpe[:, idx, :]    



seeds = [1432, 42, 8378]

net = "FCN"
log_dir = "logging_accuracy" 
dataset = "cifar_binarized"

outfile_dd = glob.glob(f"{log_dir}/{dataset}/PvU_{net}*")
outfile_nnPU = glob.glob(f"{log_dir}/{dataset}/nnPU_{net}*")
outfile_uPU = glob.glob(f"{log_dir}/{dataset}/uPU_{net}*")
outfile_our = glob.glob(f"{log_dir}/{dataset}/TEDn_{net}*")
outfile_CVIR = glob.glob(f"{log_dir}/{dataset}/CVIR_{net}*")

test_TEDn = get_method_contents(outfile_our)
test_nnPU = get_method_contents(outfile_nnPU)
test_uPU = get_method_contents(outfile_uPU)
test_PvU = get_method_contents(outfile_dd)
test_CVIR = get_method_contents(outfile_CVIR)


TEDn_acc = return_acc(test_TEDn)
TEDn_mpe = return_mpe(test_TEDn)

nnPU_acc = return_acc(test_nnPU)
nnPU_mpe = return_mpe(test_nnPU)

uPU_acc = return_acc(test_uPU)
uPU_mpe = return_mpe(test_uPU)

CVIR_acc = return_acc(test_CVIR)
CVIR_mpe = return_mpe(test_CVIR)

PvU_acc = return_acc(test_PvU)
PvU_mpe = return_mpe(test_PvU)

TEDn_acc, TEDn_mpe = converged_results(TEDn_acc, TEDn_mpe)

nnPU_acc, _ = converged_results(nnPU_acc, nnPU_mpe)

CVIR_acc, _ = converged_results(CVIR_acc, CVIR_mpe)

uPU_acc, _ = oracle_results(uPU_acc, uPU_mpe, test_uPU)

PvU_acc, PvU_mpe = oracle_results(PvU_acc, PvU_mpe, test_PvU)

print(uPU_acc)

##### MPE 
print ("-------Mean-------")
print("${:.3f}$  & ${:.3f}$ & ${:.3f}$ & ${:.3f}$".format(np.mean(TEDn_mpe, axis=0)[3], np.mean(PvU_mpe, axis=0)[4], np.mean(PvU_mpe, axis=0)[5], np.mean(PvU_mpe, axis=0)[6] ))

print("-------Mean + STD ----- ")
print("${:.3f} \pm {:.3f} $  & ${:.3f} \pm {:.3f} $ & ${:.3f} \pm {:.3f}  $ & ${:.3f} \pm {:.3f} $".format(np.mean(TEDn_mpe, axis=0)[3], np.std(TEDn_mpe, axis=0)[3], np.mean(PvU_mpe, axis=0)[4], np.std(PvU_mpe, axis=0)[4], np.mean(PvU_mpe, axis=0)[5],  np.std(PvU_mpe, axis=0)[5], np.mean(PvU_mpe, axis=0)[6], np.std(PvU_mpe, axis=0)[6]))


##### ACCC 
print ("-------Mean-------")
print("${:.1f}$  & ${:.1f}$ & ${:.1f}$ & ${:.1f}$ & ${:.1f}$ & ${:.1f}$".format(np.mean(TEDn_acc, axis=0)[2], np.mean(CVIR_acc, axis=0)[2], np.mean(PvU_acc, axis=0)[2], np.mean(PvU_acc, axis=0)[3], np.mean(nnPU_acc, axis=0)[2],  np.mean(uPU_acc, axis=0)[2]))

print("-------Mean + STD ----- ")
print("${:.1f} \pm {:.2f}$  & ${:.1f} \pm {:.2f}$ & ${:.1f} \pm {:.2f}$ & ${:.1f} \pm {:.2f}$ & ${:.1f} \pm {:.2f}$ & ${:.1f} \pm {:.2f}$".format(np.mean(TEDn_acc, axis=0)[2], np.std(TEDn_acc, axis=0)[2], np.mean(CVIR_acc, axis=0)[2], np.std(CVIR_acc, axis=0)[2],  np.mean(PvU_acc, axis=0)[2],np.std(PvU_acc, axis=0)[2], np.mean(PvU_acc, axis=0)[3], np.std(PvU_acc, axis=0)[3],  np.mean(nnPU_acc, axis=0)[2], np.std(nnPU_acc, axis=0)[2],  np.mean(uPU_acc, axis=0)[2], np.std(uPU_acc, axis=0)[2]))
