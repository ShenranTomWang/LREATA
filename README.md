# ReservoirTTA

<a href="https://arxiv.org/pdf/2505.14511?"><img src="https://img.shields.io/badge/arxiv-orange"></a>

This repository contains the official Pytorch implementation of under review paper: "ReservoirTTA: Prolonged Test-time Adaptation for Evolving and Recurring Domains"

## Abstract
This paper introduces **ReservoirTTA**, a novel plug–in framework designed for prolonged test–time adaptation (TTA) in scenarios where the test domain continuously shifts over time, including cases where domains recur or evolve gradually. At its core, ReservoirTTA maintains a reservoir of domain-specialized models—an adaptive test-time model ensemble—that both detects new domains via online clustering over style features of incoming samples and routes each sample to the appropriate specialized model, and thereby enables domain-specific adaptation. This multi-model strategy overcomes key limitations of single model adaptation, such as catastrophic forgetting, inter-domain interference, and error accumulation, ensuring robust and stable performance on sustained non-stationary test distributions. Our theoretical analysis reveals key components that bound parameter variance and prevent model collapse, while our plug–in TTA module mitigates catastrophic forgetting of previously encountered domains. Extensive experiments on the classification corruption benchmarks, including ImageNet-C and CIFAR-10/100-C, as well as the Cityscapes→ACDC semantic segmentation task, covering recurring and continuously evolving domain shifts, demonstrate that ReservoirTTA significantly improves adaptation accuracy and maintains stable performance across prolonged, recurring shifts, outperforming state-of-the-art methods.

## Overview

<img src="figs/introduction.png" alt="image" style="width:auto;height:auto;">

<b>Recurring test-time adaptation scenarios. Left: </b> Visual domains can recur over time; ETA, lacking regularization, steadily degrades under these repeated shifts. <b>Right:</b> A zoom-in on the snow corruption across 20 recurrences shows that EATA remains overall stable but still exhibits error spikes on returning to the same corruption across recurrences. <b> ReservoirTTA </b> detects returning domains and reuses specialized models to preserve learned knowledge, delivering improved robustness and faster (re-)adaptation over successive recurrences.

<br>
<br>
<br>

<img src="figs/method.png" alt="image" style="width:auto;height:auto;">

<b>Overview of ReservoirTTA.</b> ReservoirTTA operates in four stages: (1) <b>Style Characterization and Domain Identification</b> extracts early convolutional features and assigns incoming test batches to a style cluster via an online clustering mechanism; (2) <b>Model Reservoir Initialization</b> adds a new model for a detected domain, initializing it with parameters that maximize prediction mutual information; (3) <b>Model Reservoir Adaptation</b> selectively adapts the most relevant model using TTA methods; and (4) <b>Model Prediction</b> is then obtained via the ensemble’s parameters.

## Prerequisites
To use the repository please use the following conda environment

```
conda update conda
conda env create -f environment.yml
conda activate reservoirtta 
```

## Run
To execute the code, use the following command:

```
python test_time.py --cfg <config_filename>
```

Replace `<config_filename>` with the appropriate configuration file located in the `cfgs` directory.

### Examples
To run experiments on CIFAR100-C with the CSC setting:

- **ETA**:
  ```
  python test_time.py --cfg cfgs/cifar100_c/Standard/eta.yaml
  ```
- **EATA**:
  ```
  python test_time.py --cfg cfgs/cifar100_c/Standard/eata.yaml
  ```
- **EATA+ReservoirTTA**:
  ```
  python test_time.py --cfg cfgs/cifar100_c/Standard/eata_reservoir.yaml
  ```

To run experiments on CIFAR10-C with the CDC setting:

- **ROID**:
  ```
  python test_time.py --cfg cfgs/cifar10_c/Standard/roid.yaml SETTING continual_cdc
  ```
- **ROID+ReservoirTTA**:
  ```
  python test_time.py --cfg cfgs/cifar10_c/Standard/roid_reservoir.yaml SETTING continual_cdc
  ```

### Parameters to Tune
The default configurations are defined in `conf.py`. Below is a detailed explanation of the key parameters you can adjust:

#### General Parameters
- **`DATA_DIR`**: Specifies the path to the data directory where datasets are stored.
- **`SETTING`**: Defines the Test-Time Adaptation (TTA) protocol. Options include:
  - `"continual"`: For the Continual Shifting Corruption (CSC) setting.
  - `"continual_cdc"`: For the Continual Domain Corruption (CDC) setting.
- **`CORRUPTION.RECUR`**: Indicates the number of recurrences for the corruption.

#### ReservoirTTA-Specific Parameters
- **`RESERVOIRTTA.MAX_NUM_MODELS`**: The maximum number of models allowed in the reservoir.
- **`RESERVOIRTTA.SIZE_OF_BUFFER`**: The size of the style reservoir buffer used for storing style features.
- **`RESERVOIRTTA.QUANTILE_THR`**: The quantile threshold used to set the new domain detector threshold.
- **`RESERVOIRTTA.ENSEMBLING`**: A boolean flag to enable or disable weight ensembling for predictions.
- **`RESERVOIRTTA.SAMPLING`**: Specifies the method used to sample features from the style reservoir.
- **`RESERVOIRTTA.INIT`**: Defines the method for initializing a new model in the reservoir.
- **`RESERVOIRTTA.STYLE_IDX`**: A list of layers from the frozen VGG model to use as style extractors.
- **`RESERVOIRTTA.STYLE_FORMAT`**: Specifies the function type used to compute the style features from the frozen VGG model.

These parameters allow you to customize the behavior of the framework to suit your specific use case or dataset.

## Thanks
Our code is derived from https://github.com/mariodoebler/test-time-adaptation. Please follow this repository to download datasets under `data` for CIFAR10-C, CIFAR100-C, ImageNet-C, and CCC.

## Cite
```
@article{vray2025reservoirtta,
  title={ReservoirTTA: Prolonged Test-time Adaptation for Evolving and Recurring Domains},
  author={Vray, Guillaume and Tomar, Devavrat and Gao, Xufeng and Thiran, Jean-Philippe and Shelhamer, Evan and Bozorgtabar, Behzad},
  journal={arXiv preprint arXiv:2505.14511},
  year={2025}
}
```
