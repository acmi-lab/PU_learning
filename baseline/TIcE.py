import numpy as np
import math
from bitarray import bitarray
import time
import heapq

import argparse


    
def main():
    parser = argparse.ArgumentParser(description='Estimate the class prior through decision tree induction.')
    parser.add_argument('data', type=str, help='Path to the data')
    parser.add_argument('labels', type=str, help='Path to the labels')
    parser.add_argument('-o', '--out', type=str, help='Path to save output to')
    parser.add_argument("-f",'--folds', default=None, type=str, help='Path to the folds, if no folds are provided, 5 random folds are chosen.')
    parser.add_argument("-d",'--delta', default=None, type=float, help='Delta, default: using formula from paper.')
    parser.add_argument("-k",'--max-bepp', default=5, type=int, help='The max-bepp parameter k, default=5.')
    parser.add_argument("-M",'--maxSplits', default=500, type=int, help='The maximum number of splits in the decision tree, default=500.')
    parser.add_argument('--promis', action="store_true", help="Set this option to only use the most promising subset (instead of calculating the maximum lower bound)")
    parser.add_argument('--delimiter', default=',', type=str, help="Delimiter of the data files")
    parser.add_argument("-m",'--minT', default=10, type=int, help='The minimum set size to update the lower bound with, default=10.')
    parser.add_argument("-i", '--nbIts', default=2, type=int, help="The number of times to repeat the the estimation process. Default 2 (first with c_prior=0.5, then with c_prior=c_estimate)")
    
    args = parser.parse_args()
      
    data = np.genfromtxt(args.data, delimiter=args.delimiter)
    labels = np.genfromtxt(args.labels, delimiter=args.delimiter)
    labels = bitarray(list(labels==1))
    folds = np.array(map(lambda l: int(l.strip()), open(args.folds).readlines())) if args.folds else np.random.randint(5, size=len(data))
    
    ti = time.time()    
    (c_estimate, c_its_estimates) = tice(data, labels, args.max_bepp, folds, args.delta,nbIterations=args.nbIts, maxSplits=args.maxSplits, useMostPromisingOnly=args.promis,minT=args.minT)
    ti = time.time() - ti
    
    
    alpha=1.0
    if c_estimate > 0:
      pos = float(labels.count())/c_estimate
      tot = len(data)
      alpha=max(0.0,min(1.0,pos/tot))


    print("c:"+str(c_estimate))
    print("alpha:"+str(alpha))
    
    # Write output
    if args.out:
        outfile = open(args.out,'w+')
        for arg in vars(args):
            outfile.write(arg+":\t"+str(vars(args)[arg])+"\n")
        outfile.write("\n")
        for (it,c_estimates) in zip(range(1, args.nbIts+1), c_its_estimates):
            outfile.write("c_estimates it "+str(it)+":\t"+str(c_estimates)+"\n")
        outfile.write("\n")
        outfile.write("c_estimate:\t"+str(c_estimate)+"\n")
        outfile.write("alpha_estimate:\t"+str(alpha)+"\n")
        outfile.write("time:\t"+str(ti)+"\n")
            
        outfile.flush()
        outfile.close()
        


def pick_delta(T):
  return max(0.025, 1/(1+0.004*T))
  
  
  
def low_c(data, label, delta, minT,c=0.5):
  T = float(data.count())
  if T<minT:
    return 0.0
  # print(data)
  # import pdb; pdb.set_trace()
  # print(label)
  # print(bitarray(label))
  L = float((data & bitarray(label)).count())
  clow = L/T - math.sqrt(c*(1-c)*(1-delta)/(delta*T))
  return clow

def max_bepp(k):
    def fun(counts):
        return max(list(map(lambda T_P: (0 if T_P[0] == 0 else float(T_P[1]) / (T_P[0] + k)), counts)))
    return fun

def generate_folds(folds):
    for fold in range(max(folds)+1):
        tree_train = bitarray(list(folds==fold))
        estimate = ~tree_train
        yield (tree_train, estimate)
        
        
def tice(data, labels, k, folds, delta=None, nbIterations=2, maxSplits=500, useMostPromisingOnly=False, minT=10, ):
    
    if isinstance(labels, np.ndarray):
        labels = bitarray(list(labels == 1))

    c_its_ests = []
    c_estimate = 0.5
    
    
    for it in range(nbIterations):
        print(it)
        c_estimates = []
    
    
        global c_cur_best # global so that it can be used for optimizing queue.
        for (tree_train, estimate) in generate_folds(folds):
            c_cur_best = low_c(estimate, labels, 1.0, minT, c=c_estimate)
            cur_delta = delta if delta else pick_delta(estimate.count())
            
            
            if useMostPromisingOnly:
                
                c_tree_best=0.0
                most_promising = estimate
                for tree_subset, estimate_subset in subsetsThroughDT(data, tree_train, estimate, labels, splitCrit=max_bepp(k), minExamples=minT, maxSplits=maxSplits, c_prior=c_estimate, delta=cur_delta):
                    tree_est_here = low_c(tree_subset,labels,cur_delta, 1,c=c_estimate)
                    if tree_est_here > c_tree_best:
                        c_tree_best = tree_est_here
                        most_promising = estimate_subset
                        
                c_estimates.append(max(c_cur_best, low_c(most_promising, labels, cur_delta,minT, c=c_estimate)))
                
            else:
                
                for tree_subset, estimate_subset in subsetsThroughDT(data, tree_train, estimate, labels, splitCrit=max_bepp(k), minExamples=minT, maxSplits=maxSplits, c_prior=c_estimate, delta=cur_delta):
                    est_here = low_c(estimate_subset,labels,cur_delta, minT,c=c_estimate)
                    c_cur_best=max(c_cur_best, est_here)
                c_estimates.append(c_cur_best)
                
            
        c_estimate = sum(c_estimates)/float(len(c_estimates))
        c_its_ests.append(c_estimates)
        
    return c_estimate, c_its_ests




def subsetsThroughDT(data, tree_train, estimate, labels, splitCrit=max_bepp(5), minExamples=10, maxSplits=500,
                     c_prior=0.5, delta=0.0, n_splits=3):
  # This learns a decision tree and updates the label frequency lower bound for every tried split.
  # It splits every variable into 4 pieces: [0,.25[ , [.25, .5[ , [.5,.75[ , [.75,1]
  # The input data is expected to have only binary or continues variables with values between 0 and 1.
  # To achieve this, the multivalued variables should be binarized and the continuous variables should be normalized
  
  # Max: Return all the subsets encountered
  
  all_data = tree_train | estimate
  
  borders = np.linspace(0, 1, n_splits + 2, True).tolist()[1: -1]
  
  def makeSubsets(a):
      subsets = []
      options = bitarray(all_data)
      for b in borders:
          X_cond = bitarray(list((data[:, a] < b))) & options
          options &= ~X_cond
          subsets.append(X_cond)
      subsets.append(options)
      return subsets
      
  conditionSets = [makeSubsets(a) for a in range(data.shape[1])]
  
  priorityq = []
  heapq.heappush(priorityq, (-low_c(tree_train, labels, delta, 0, c=c_prior), -(tree_train&labels).count(), tree_train,
                             estimate, set(range(data.shape[1])), 0))
  yield (tree_train, estimate)
  
  n = 0
  minimumLabeled = 1
  while n < maxSplits and len(priorityq) > 0:
    n += 1
    (ppos, neg_lab_count, subset_train, subset_estimate, available, depth) = heapq.heappop(priorityq)
    lab_count = -neg_lab_count
    
    best_a = -1
    best_score = -1
    best_subsets_train = []
    best_subsets_estimate = []
    best_lab_counts = []
    uselessAs = set()
    
    for a in available:
      subsets_train = list(map(lambda X_cond: X_cond & subset_train, conditionSets[a]))
      subsets_estimate = list(map(lambda X_cond: X_cond & subset_estimate, conditionSets[a]))  # X_cond & subset_train
      estimate_lab_counts = list(map(lambda subset: (subset & labels).count(), subsets_estimate))
      if max(estimate_lab_counts) < minimumLabeled:
        uselessAs.add(a)
      else:
        score = splitCrit(list(map(lambda subsub: (subsub.count(), (subsub & labels).count()), subsets_train)))
        if score > best_score:
            best_score = score
            best_a = a
            best_subsets_train = subsets_train
            best_subsets_estimate = subsets_estimate
            best_lab_counts = estimate_lab_counts

    fake_split = len(list(filter(lambda subset: subset.count() > 0, best_subsets_estimate))) == 1
    
    if best_score > 0 and not fake_split:
      newAvailable = available - {best_a} - uselessAs
      for subsub_train, subsub_estimate in zip(best_subsets_train, best_subsets_estimate):
        yield (subsub_train, subsub_estimate)
      minimumLabeled = c_prior * (1 - c_prior) * (1 - delta) / (delta * (1 - c_cur_best) ** 2)
          
      for (subsub_lab_count, subsub_train, subsub_estimate) in zip(best_lab_counts, best_subsets_train,
                                                                   best_subsets_estimate):
          if subsub_lab_count > minimumLabeled:
            total = subsub_train.count()
            if total > minExamples:  # stop criterion: minimum size for splitting
              train_lab_count = (subsub_train & labels).count()
              if lab_count != 0 and lab_count != total:  # stop criterion: purity
                heapq.heappush(priorityq, (-low_c(subsub_train, labels, delta, 0, c=c_prior), -train_lab_count,
                                           subsub_train, subsub_estimate, newAvailable, depth+1))  
  

if __name__=='__main__':
    main()
