## Model Architecture

                                   FatEdgeCentricGNN Model Architecture

                                                (Input)
                                          (x, edge_index, edge_attr, batch)
                                                |
              +-------------------------------------------------+
              |                                                 |
              v                                                 v
        [Node Encoder]                              [Edge Encoder]
           (x_encoded)                                (edge_attr_encoded)
              |                                                 |
              +-------------------------------------------------+
                                     |
                                     v
+---------------------------------------------------------------------------------------------------------------------------------+
|                                                           GNN Layers (Nodes & Edges)                                            |
+---------------------------------------------------------------------------------------------------------------------------------+
                                     |
                                     | (x_encoded, edge_index, edge_attr_encoded)
                                     v
                             +-----------------------+
                             | Block 2 (GINE Sequence) |
                             +-----------------------+
                                     |
                                     |  x_current (x_init for Block 2)
                                     |  -----------------------+
                                     v                        |
               [GINEConv 2_a] --(LeakyReLU, Dropout)--> [BatchNorm 2_a]
                     |                                       |
                     v                                       |
               [GINEConv 2_b] --(LeakyReLU, Dropout)--> [BatchNorm 2_b]
                     |                                       |
                     v                                       |
               [GINEConv 2_c] --(LeakyReLU, Dropout)--> [BatchNorm 3_a] *Note: `bn3_a` used here
                     |                                       |
                     v                                       |
                (Add with skip_2(x_init))<------------------+----- [skip_2(x_init)]
                     |                                       |
                     v                                       |
               [BatchNorm 2] (Post-skip BN for Block 2)
                     |
                     v
             +-----------------------+
             | Mid Layer 1 (Transformer) |
             +-----------------------+
                     |
                     v
       [TransformerConv (conv_mid)] --(LeakyReLU, Dropout)--> [BatchNorm (bn_mid)]
                     |
                     | (x_current becomes x_init for Block 3)
                     | -----------------------+
                     v                        |
             +-----------------------+        |
             | Block 3 (GINE Sequence) |        |
             +-----------------------+        |
                     |                        |
                     v                        |
               [GINEConv 3_a] --(LeakyReLU, Dropout)--> [BatchNorm 3_a] *Note: `bn3_a` reused
                     |                                       |
                     v                                       |
               [GINEConv 3_b] --(LeakyReLU, Dropout)--> [BatchNorm 3_b]
                     |                                       |
                     v                                       |
               [GINEConv 3_c] --(LeakyReLU, Dropout)--> [BatchNorm 3_c]
                     |                                       |
                     v                                       |
                (Add with skip_3(x_init))<------------------+----- [skip_3(x_init)]
                     |                                       |
                     v                                       |
               [BatchNorm 2] (Reused Post-skip BN for Block 3)
                     |
                     v
             +-----------------------+
             | Mid Layer 2 (Transformer) |
             +-----------------------+
                     |
                     v
     [TransformerConv (conv_mid2)] --(LeakyReLU, Dropout)--> [BatchNorm (bn_mid2)]
                     |
                     | (x_current becomes x_init for Block 4)
                     | -----------------------+
                     v                        |
             +-----------------------+        |
             | Block 4 (GINE Sequence) |        |
             +-----------------------+        |
                     |                        |
                     v                        |
               [GINEConv 4_a] --(LeakyReLU, Dropout)--> [BatchNorm 4_a]
                     |                                       |
                     v                                       |
               [GINEConv 4_b] --(LeakyReLU, Dropout)--> [BatchNorm 4_b]
                     |                                       |
                     v                                       |
               [GINEConv 4_c] --(LeakyReLU, Dropout)--------+ (No BN directly after)
                     |                                       |
                     v                                       |
                (Add with skip_4(x_init))<------------------+----- [skip_4(x_init)]
                     |                                       |
                     v                                       |
               [BatchNorm 4_c] (Post-skip BN for Block 4)
                     |
                     | (x_current becomes x_init for Block 5)
                     | -----------------------+
                     v                        |
             +-----------------------+        |
             | Block 5 (GINE Sequence) |        |
             +-----------------------+        |
                     |                        |
                     v                        |
               [GINEConv 5_A] --(LeakyReLU, Dropout)--> [BatchNorm 5_a]
                     |                                       |
                     v                                       |
               [GINEConv 5_B] --(LeakyReLU, Dropout)--> [BatchNorm 5_b]
                     |                                       |
                     v                                       |
                (Add with skip_5(x_init))<------------------+----- [skip_5(x_init)]
                     |                                       |
                     v                                       |
               [BatchNorm 5] (Post-skip BN for Block 5)
                     |
                     v
             +-----------------------+
             | Final Layer (Transformer) |
             +-----------------------+
                     |
                     v
     [TransformerConv (conv_final)] --(Dropout, LeakyReLU)--> [BatchNorm (bn_final)]
                     |
+---------------------------------------------------------------------------------------------------------------------------------+
|                                                       Readout & Classification                                                |
+---------------------------------------------------------------------------------------------------------------------------------+
                     |
                     v
              [Global Mean Pool] (using 'batch' tensor)
                     |
                     v
           [Fully Connected Layers (fc)]
             (Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear)
                     |
                     v
               (Output Logits)



## Method Description


Essencially is uses 3 "big Blocks" of 3 GINEconv layers with dropout, batch regularizations, and several residual connections. 
Moreover after each bigBlock there is one block of Trasformer convolutions with a limited amout of heads. 
The combinations of the two allows the net to learn well the relations between nodes and edges but not too well in order to avoid noise overfitting.


## File Structure

```
.
├── main.py  
├── models.py                 # Main training and inference script
├── checkpoints/              # Model checkpoints
│   ├── model_A_epoch_*.pth
│   ├── model_B_epoch_*.pth
│   ├── model_C_epoch_*.pth
│   └── model_D_epoch_*.pth
├── logs/                     # Training logs
│   ├── log_A.log
│   ├── log_B.log
│   ├── log_C.log
│   └── log_D.log
├── submission/               # Prediction outputs
│   ├── testset_A.csv
│   ├── testset_B.csv
│   ├── testset_C.csv
│   └── testset_D.csv
├── requirements.txt          # Dependencies
├── submissionTool.ipynb
└── README.md                # This file
```

## Usage

For Database D use : 

node_dim = 1 
    hidden_dim = 64
    output_dim = 6
    edge_dim = 7  
    dropout = 0.2
    learning_rate = 0.005
    num_epochs = 100
    batch_size = 64
    seed = 420
    checkpoint for inference = model_D_best which is the same of model_D_epoch_69
    loss = NoisyCrossEntropyLoss(p_noisy=0.4) 


For Database C use : 

node_dim = 1 
    hidden_dim = 64
    output_dim = 6
    edge_dim = 7  
    dropout = 0.2
    learning_rate = 0.005
    num_epochs = 100
    batch_size = 64
    seed = 420
    checkpoint for inference = model_C_best which is the same of model_C_epoch_98
    loss = NoisyCrossEntropyLoss(p_noisy=0.2) 


For Database B use : 

node_dim = 1 
    hidden_dim = 64
    output_dim = 6
    edge_dim = 7  
    dropout = 0.2
    learning_rate = 0.005
    num_epochs = 100
    batch_size = 64
    seed = 420
    checkpoint for inference = model_B_best which is the same of model_B_epoch_72
    loss = NoisyCrossEntropyLoss(p_noisy=0.2) 

For Database A use : 

node_dim = 1 
    hidden_dim = 64
    output_dim = 6
    edge_dim = 7  
    dropout = 0.3
    learning_rate = 0.005
    num_epochs = 100
    batch_size = 64
    seed = 420
    checkpoint for inference = model_B_best which is the same of model_B_epoch_84
    loss = NoisyCrossEntropyLoss(p_noisy=0.2) 





