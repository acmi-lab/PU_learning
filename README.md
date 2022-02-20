
# Mixture Proportion Estimation and PU Learning: A Modern Approach

This repository is the official implementation of [Mixture Proportion Estimation and PU Learning: A Modern Approach](https://arxiv.org/abs/2111.00980). We also release implementation of relevant baselines for [MPE](https://raw.githubusercontent.com/acmi-lab/PU_learning/main/files/MPE.txt) and [PU learning](https://raw.githubusercontent.com/acmi-lab/PU_learning/main/files/PU_classification.txt). 
If you find this repository useful or use this code in your research, please cite the following paper: 

> Garg, S., Wu, Y., Smola, A., Balakrishnan, S., Lipton, Z. (2021). Mixture Proportion Estimation and PU Learning: A Modern Approach. arxiv preprint  arXiv:2111.00980. 
```
@article{garg2021mixture,
    title={Mixture Proportion Estimation and PU Learning: A Modern Approach},
    author={Garg, Saurabh and Wu, Yifan and Smola, Alex and Balakrishnan, Sivaraman and Lipton, Zachary C.},
    year={2021},
    journal={arXiv preprint arXiv:2111.00980},
}
```

## Requirements

The code is written in Python and uses [PyTorch](https://pytorch.org/). To install requirements, setup a conda enviornment using the following command:

```setup
conda create --file requirements.txt
```

## Quick Experiments 

`train.py` file is the main entry point for training the model and run the code with the following command:

```setup
python train_PU.py --data-type="cifar_DogCat" --train-method="TEDn" --net-type="ResNet" --epochs=1000 --warm-start --warm-start-epochs=100 --alpha=0.5
```

Change the parameters to your liking and run the experiment. For example, change dataset with varying --data-type and vary algorithm with varying --train-method. We implement the BBE estimator in `estimator.py` and CVIR algorithm in `algorithm.py`.

## Scripts 
We provide a set of scripts to run experiments. See scripts folder for details. We provide paper results in paper_results folder and the corresponding code for plots in plot_helper. 

## License
This repository is licensed under the terms of the [Apache-2.0 License](LICENSE).

## Questions?

For more details, refer to the accompanying NeurIPS 2021 paper (Spotlight): [Mixture Proportion Estimation and PU Learning: A Modern Approach](https://arxiv.org/abs/2111.00980). If you have questions, please feel free to reach us at sgarg2@andrew.cmu.edu or open an issue.  
