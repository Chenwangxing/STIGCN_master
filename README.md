# STIGCN_master
STIGCN: Spatial-Temporal Interaction-aware Graph Convolution Network for Pedestrian Trajectory Prediction

Code will be released soon!

The general framework of the proposed method. First, historical trajectories were transformed into spatial and temporal graph inputs. Then, spatial-temporal interaction-aware learning obtained the spatial-temporal fusion adjacency matrix from the graph inputs. Afterward, the subsequent graph convolution network learned the trajectory representation features. Finally, the Time-Extrapolator Pyramid Convolution Neural Network (TEP-CNN) estimated the bi-variate Gaussian distribution parameters of future trajectory points for predicting future pedestrian trajectories.

![image](https://github.com/Chenwangxing/STIGCN_master/assets/72364851/e26fd25e-e797-4d62-aa84-2a85ccf3530a)

The spatial-temporal interaction-aware learning framework. First, embedding functions were used to obtain spatial and temporal graph inputs that represent features of the graph. Then, the spatial and temporal adjacency matrices were generated through the self-attention mechanism. Next, the spatial-temporal interaction-aware attention module further learns the relationship between spatial and temporal interactions to generate the spatial-temporal awareness adjacency matrix. Finally, the spatial-temporal adjacency matrix and spatial-temporal interaction-aware adjacency matrix were concatenated to generate the spatial-temporal fusion adjacency matrix.

![image](https://github.com/Chenwangxing/STIGCN_master/assets/72364851/a81c7faf-0340-4f26-adbd-be9fde75c172)



Some codes are borrowed from Social-STGCNN and SGCN. We gratefully acknowledge the authors for posting their code.
