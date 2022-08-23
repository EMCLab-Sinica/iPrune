# Intermittent-Aware Neural Network Pruning

<!-- ABOUT THE PROJECT -->
## Overview

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup and Build](#setup-and-build)

## Directory/File Structure
Below is an explanation of the directories/files found in this repository.

`pruning/datasets` contains the datasets used for three models.<br/>
`pruning/models` contains the model information for training.<br/>
`pruning/onnx_models` contains the onnx models deployed on TI-MSP430FR5994.<br/>
`pruning/pruning_utils` contains auxiliary functions used in the network pruning.<br/>
`pruning/main.py` is the entry file in the intermittent-aware neural network pruning.<br/>
`pruning/config.py` contains the model configuration and tile size used during runtime inference.<br/>
`pruning/prune.squeezenet.sh`, `pruning/prune.har.sh`, and `pruning/prune.kws_cnn_s.sh` are the scripts that run the intermittent-aware neural network pruning.<br/>
`inference-library` contains the both inference runtime library designed for intermittently-powered systems and coutinuously-powered systems (currently supports convolution, sparse convolution, fully connected layers, sparse fully connected layers, global pooling, and batchnormalization layers).<br/>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

###### Intermittent-Aware Neural Network Pruning
Here are basic software to build the intermittent-aware neural network pruning
* Python >= 3.7
* Several deep learning Python libraries defined in `pruning/requirements.txt`. Those libraries can be installed with `pip3 install -r requirements.txt`.

###### Intermittent Inference Library
Here is the basic software and hardware needed to build/run the intermittent inference runtime library.
* Python >= 3.7
* Several deep learning Python libraries defined in `inference-library/requirements.txt`. Those libraries can be installed with `pip3 install -r requirements.txt`.
* [Code composer studio](https://www.ti.com/tool/CCSTUDIO) >= 11.0
* [MSP-EXP430FR5994 LaunchPad](https://www.ti.com/tool/MSP-EXP430FR5994)
* [MSP DSP Library](https://www.ti.com/tool/MSP-DSPLIB) 1.30.00.02
* [MSP430 driverlib](https://www.ti.com/tool/MSPDRIVERLIB) 2.91.13.01

### Setup and Build

###### Intermittent-Aware Neural Network Pruning
1. Download/clone this repository
2. Install dependencies `pip3 install -r requirements.txt`
3. Run intermittent-aware neural network pruning scripts: `source prune.kws_cnn_s.sh`

###### Intermittent Inference Library
1. Download/clone this repository
2. Install dependencies `pip3 install -r requirements.txt`
3. Convert the provided pre-trained models with the command `cd inference-library && python transform.py --target msp430 (--hawaii|--baseline --stable-power) (pruned_cifar10|pruned_har|pruned_kws_cnn) --method (intermittent|energy)` to specify the target platform, the inference engine, the model, and pruning method to deploy.
4. Download [MSP DSP Library](https://www.ti.com/tool/MSP-DSPLIB) to `inference-library/TI-DSPLib` and apply the patch with the following command:
```
cd TI-DSPLib/ && patch -Np1 -i ../TI-DSPLib.diff
```
5. Download `MSP430 driverlib` to `inference-library/msp430/` with the following command:
```
cd inference-library/msp430/ && git clone ssh://git@github.com/EMCLab-Sinica/driverlib-msp430.git driverlib
```
6. Import the folder `inference-library/msp430/` as a project in CCSTUDIO
