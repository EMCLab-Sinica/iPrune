# Intermittent-Aware Neural Network Pruning

<!-- ABOUT THE PROJECT -->
## Overview

Deep neural network inference on energy harvesting tiny devices has emerged as a solution for sustainable edge intelligence. However, compact models optimized for continuously-powered systems may become suboptimal when deployed on intermittently-powered systems. This paper presents the pruning criterion, pruning strategy, and prototype implementation of iPrune, the first framework which introduces intermittency into neural network pruning to produce compact models adaptable to intermittent systems. The pruned models are deployed and evaluated on a Texas Instruments device with various power strengths and TinyML applications.
Compared to an energy-aware pruning framework, iPrune can speed up intermittent inference by 1.1 to 2 times while achieving comparable model accuracy.

Demo video: https://youtu.be/Dzg46_MO66w <!-- also send the demo link to Prof. Hsiu -->

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup and Build](#setup-and-build)

## Directory/File Structure
Below is an explanation of the directories/files found in this repository.

* `pruning/datasets/` contains the datasets used for three models.
* `pruning/models/` contains the model information for training.
* `pruning/onnx_models/` contains the onnx models deployed on TI-MSP430FR5994.
* `pruning/pruning_utils/` contains auxiliary functions used in the network pruning.
* `pruning/main.py` is the entry file in the intermittent-aware neural network pruning.
* `pruning/config.py` contains the model configuration and tile size used during runtime inference.
* `pruning/prune.squeezenet.sh`, `pruning/prune.har.sh`, and `pruning/prune.kws_cnn_s.sh` are the scripts that run the intermittent-aware neural network pruning.
* `inference-library/` contains the both inference runtime library designed for intermittently-powered systems and coutinuously-powered systems (currently supports convolution, sparse convolution, fully connected layers, sparse fully connected layers, max pooling, global average pooling, and batch normalization layers).


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

###### Intermittent-Aware Neural Network Pruning
Here are basic software to build the intermittent-aware neural network pruning
* Python 3.7
* Several deep learning Python libraries defined in `inference-library/requirements.txt`. Those libraries can be installed with `python3.7 -m pip install -r requirements.txt`.

###### Intermittent Inference Library
Here is the basic software and hardware needed to build/run the intermittent inference runtime library.
* [Code composer studio](https://www.ti.com/tool/CCSTUDIO) >= 11.0
* [MSP-EXP430FR5994 LaunchPad](https://www.ti.com/tool/MSP-EXP430FR5994)
* [MSP DSP Library](https://www.ti.com/tool/MSP-DSPLIB) 1.30.00.02
* [MSP430 driverlib](https://www.ti.com/tool/MSPDRIVERLIB) 2.91.13.01

### Setup and Build

###### Intermittent-Aware Neural Network Pruning
1. Download/clone this repository
2. Install dependencies `python3.7 -m pip install -r inference-library/requirements.txt`
3. Run intermittent-aware neural network pruning scripts: `bash prune.kws_cnn_s.sh`

###### Intermittent Inference Library
1. Download/clone this repository
3. Convert the provided pre-trained models with the command `cd inference-library && python3.7 transform.py --target msp430 --hawaii (pruned_cifar10|pruned_har|pruned_kws_cnn) --method (intermittent|energy) --sparse` to specify the target platform, the inference engine, the model, and pruning method to deploy.
4. Download and extract [MSP DSP Library](https://www.ti.com/tool/MSP-DSPLIB) to `inference-library/TI-DSPLib` and apply the patch with the following command:
```
cd TI-DSPLib/ && patch -Np1 -i ../TI-DSPLib.diff
```
5. Download and extract [MSP430 driverlib](https://www.ti.com/tool/MSPDRIVERLIB), and copy `driverlib/MSP430FR5xx_6xx` folder into the `inference-library/msp430/` folder.
6. Import the folder `inference-library/msp430/` as a project in CCSTUDIO
