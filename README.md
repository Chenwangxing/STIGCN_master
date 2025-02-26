# STIGCN_master
STIGCN: Spatial-Temporal Interaction-aware Graph Convolution Network for Pedestrian Trajectory Prediction!
The Paper: https://link.springer.com/article/10.1007/s11227-023-05850-8

The code and weights have been released, enjoy it！

The general framework of the proposed method. First, historical trajectories were transformed into spatial and temporal graph inputs. Then, spatial-temporal interaction-aware learning obtained the spatial-temporal fusion adjacency matrix from the graph inputs. Afterward, the subsequent graph convolution network learned the trajectory representation features. Finally, the Time-Extrapolator Pyramid Convolution Neural Network (TEP-CNN) estimated the bi-variate Gaussian distribution parameters of future trajectory points for predicting future pedestrian trajectories.

![image](https://github.com/Chenwangxing/STIGCN_master/assets/72364851/e26fd25e-e797-4d62-aa84-2a85ccf3530a)

The spatial-temporal interaction-aware learning framework. First, embedding functions were used to obtain spatial and temporal graph inputs that represent features of the graph. Then, the spatial and temporal adjacency matrices were generated through the self-attention mechanism. Next, the spatial-temporal interaction-aware attention module further learns the relationship between spatial and temporal interactions to generate the spatial-temporal awareness adjacency matrix. Finally, the spatial-temporal adjacency matrix and spatial-temporal interaction-aware adjacency matrix were concatenated to generate the spatial-temporal fusion adjacency matrix.

![image](https://github.com/Chenwangxing/STIGCN_master/assets/72364851/a81c7faf-0340-4f26-adbd-be9fde75c172)


## Code Structure
checkpoint folder: contains the trained models

dataset folder: contains ETH and UCY datasets

model.py: the code of STIGCN

train.py: for training the code

test.py: for testing the code

utils.py: general utils used by the code

metrics.py: Measuring tools used by the code

## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test.py


## Trajectory prediction update
Different from previous random sampling (MC), we introduce Latin hypercube sampling (LHS) in pedestrian trajectory prediction to mitigate the long-tail effect. Compared with quasi-Monte Carlo sampling (QMC), Latin hypercube sampling is more suitable for trajectory prediction and can more accurately describe the diversity of pedestrian motion. It is worth noting that random sampling, quasi-Monte Carlo sampling, and Latin hypercube sampling are plug-and-play and do not require training. （For details, please refer to the paper: DSTIGCN: Deformable Spatial-Temporal Interaction Graph Convolution Network for Pedestrian Trajectory Prediction）

Prediction diagram of each sampling method. The top is a twodimensional scatter plot of 20 points using MC, QMC and LHS, respectively.
The asterisks represent the coordinates of the true destination in the sampling
space; the bottom is 20 random trajectories predicted by each method.
<img width="955" alt="不同采样方法的示意图 - 修改1" src="https://github.com/user-attachments/assets/cb0bd0ef-e9b2-4646-9d05-4417ca399b01" />

You can easily run the model! To use QMC sampling please run:  test-Qmc.py

You can easily run the model! To use LHS sampling please run:  test-Lhs.py

The prediction errors of different sampling methods are shown in the following table：
| STIGCN  | ETH | HOTEL| UNIV| ZARA1 | ZARA2 | AVG |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| MC  | 0.58/0.96 | 0.30/0.44| 0.38/0.67| 0.28/0.47 | 0.23/0.42 | 0.35/0.59 |
| QMC  | 0.52/0.96 | 0.22/0.33| 0.31/0.56| 0.25/0.45 | 0.21/0.39 | 0.30/0.54 |
| LHS  | 0.43/0.68 | 0.24/0.48| 0.26/0.48| 0.22/0.41 | 0.17/0.32 | 0.26/0.47 |

## Acknowledgement
Some codes are borrowed from Social-STGCNN and SGCN. We gratefully acknowledge the authors for posting their code.


## Cite this article:

Chen, W., Sang, H., Wang, J. et al. STIGCN: spatial–temporal interaction-aware graph convolution network for pedestrian trajectory prediction. J Supercomput (2023). https://doi.org/10.1007/s11227-023-05850-8

@article{chen2024stigcn,
  title={STIGCN: spatial--temporal interaction-aware graph convolution network for pedestrian trajectory prediction},
  author={Chen, Wangxing and Sang, Haifeng and Wang, Jinyu and Zhao, Zishan},
  journal={The Journal of Supercomputing},
  volume={80},
  number={8},
  pages={10695--10719},
  year={2024},
  publisher={Springer}
}
