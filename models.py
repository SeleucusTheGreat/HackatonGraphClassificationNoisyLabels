import argparse
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm 
from loadData import GraphDataset
import os
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import f1_score 
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import logging
from datetime import datetime


class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()
    
def add_zeros(data):
    data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
    return data

class FatEdgeCentricGNN(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, edge_dim, dropout=0.2):
        super().__init__()
        
        # Node and edge feature encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )


        # GINEConv layers with edge feature support
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ), edge_dim=hidden_dim, train_eps= True) 
        
        self.conv2_a = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True) 
        
        self.conv2_b = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True) 
        
        self.conv2_c = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True) 
        
        self.conv3_a = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True) 
        
        self.conv3_b = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True) 
              

        
        self.conv3_c = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True)

        self.conv4_a = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True)

        self.conv4_b = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True)

        self.conv4_c = GINEConv(    
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True)
        
        self.conv5_A = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True)
        
        self.conv5_B = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), edge_dim=hidden_dim, train_eps= True)



              
        

        heads = 8
        self.conv_final = TransformerConv(
            hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout, bias=True
        )
        self.conv_mid = TransformerConv(
            hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout, bias=True
        )
        self.conv_mid2 = TransformerConv (
            hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout, bias=True
        )




       
        # Normalization and dropout
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2_a = BatchNorm(hidden_dim)
        self.bn2_b = BatchNorm(hidden_dim)
        self.bn2_c = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3_a = BatchNorm(hidden_dim)
        self.bn3_b = BatchNorm(hidden_dim)
        self.bn3_c = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.bn4_a = BatchNorm(hidden_dim)
        self.bn4_b = BatchNorm(hidden_dim)
        self.bn4_c = BatchNorm(hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)
        self.bn5_a = BatchNorm(hidden_dim)
        self.bn5_b = BatchNorm(hidden_dim)
        self.bn5 = BatchNorm(hidden_dim)
        self.bn_final = BatchNorm(hidden_dim)
        self.bn_mid = BatchNorm(hidden_dim)
        self.bn_mid2 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Readout and classification
        self.pool = global_mean_pool
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim),
        )
        # Skip connections
        self.skip_1 = nn.Linear(hidden_dim, hidden_dim)
        self.skip_2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip_3 = nn.Linear(hidden_dim, hidden_dim)
        self.skip_4 = nn.Linear(hidden_dim, hidden_dim)
        self.skip_5 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Layer 1 Encode edge features and node features
        
        edge_attr = self.edge_encoder(edge_attr)
        x = self.node_encoder(x)


        # Layer 2 GINEConv with edge features
        x_init = x.clone()
        
        x = self.conv2_a(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn2_a(x)

        x = self.conv2_b(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn2_b(x)

        x = self.conv2_c(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn3_a(x)
        
        x = x + self.skip_2(x_init)  
        x = self.bn2(x)

        # Layer 3 GINEConv with edge features
        # Skip connection
        x = self.conv_mid(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn_mid(x)

       
        x_init = x.clone()
        
        x = self.conv3_a(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn3_a(x)

        x = self.conv3_b(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn3_b(x)

        x = self.conv3_c(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn3_c(x)
        
        x = x + self.skip_3(x_init)  
        x = self.bn2(x)

        # Skip connection for mid layer
        x = self.conv_mid2(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn_mid2(x)

        
        # Layer 3 GINEConv with edge features
        x_init = x.clone()
        
        x = self.conv4_a(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        x = self.bn4_a(x)
        x = self.conv4_b(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        x = self.bn4_b(x)
        x = self.conv4_c(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        
        x = x + self.skip_4(x_init)
        x = self.bn4_c(x)


        # Layer 4 GINEConv with edge features
        x_init = x.clone()

        x = self.conv5_A(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn5_a(x)

        
        x = self.conv5_B(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn5_b(x)

        x = x + self.skip_5(x_init)
        x = self.bn5(x)


        # Layer final with skip connection with attention
        x = self.conv_final(x, edge_index, edge_attr=edge_attr)
        x = self.dropout(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn_final(x)
        
        # FFN + classification
        x = self.pool(x, batch)
        x = self.fc(x)
        return x
    
class EdgeCentricGNN(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, edge_dim, dropout=0.2):
        super().__init__()
        
        # Node and edge feature encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        # GINEConv layers (your existing definitions are fine here)
        # ... (conv1, conv2_a, conv2_b, conv2_c, conv3_a, conv3_b, conv3_c, conv4_a, conv4_b, conv4_c) ...
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)), 
            edge_dim=hidden_dim, train_eps= True) 
        self.conv2_a = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), 
            edge_dim=hidden_dim, train_eps= True) 
        self.conv2_b = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), 
            edge_dim=hidden_dim, train_eps= True) 
       
        self.conv3_a = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), 
            edge_dim=hidden_dim, train_eps= True) 
        self.conv3_b = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), 
            edge_dim=hidden_dim, train_eps= True) 
        

        heads = 8
        self.conv_final = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout, bias=True)
        self.conv_mid = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout, bias=True)
        self.conv_mid2 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout, bias=True)

        # Normalization and dropout - Corrected
        # Note: BatchNorm from torch_geometric.nn is typically BatchNorm1d equivalent for node features.
        # LayerNorm from torch.nn.
        
        # BN1 is unused, removed for now, add if needed for conv1
        self.bn2_a = nn.LayerNorm(hidden_dim) 
        self.bn2_b = nn.LayerNorm(hidden_dim) 
        self.bn2_c = nn.LayerNorm(hidden_dim) 
        self.bn2_block = BatchNorm(hidden_dim) 

        self.bn3_a = nn.LayerNorm(hidden_dim) 
        self.bn3_b = nn.LayerNorm(hidden_dim)
        self.bn3_c = nn.LayerNorm(hidden_dim) 
        self.bn3_block = BatchNorm(hidden_dim) 

        self.bn4_a = nn.LayerNorm(hidden_dim) 
        self.bn4_b = nn.LayerNorm(hidden_dim) 
        self.bn4_c = nn.LayerNorm(hidden_dim) 
        self.bn4_block = BatchNorm(hidden_dim) 

        self.bn_final = BatchNorm(hidden_dim)
        self.bn_mid = nn.LayerNorm(hidden_dim) 
        self.bn_mid2 = nn.LayerNorm(hidden_dim) 
        self.bn_mid3 = nn.LayerNorm(hidden_dim) 

        self.dropout = nn.Dropout(dropout)
        
        # Readout and classification
        self.pool = global_mean_pool
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim),
        )
        # Skip connections
        self.skip_1 = nn.Linear(hidden_dim, hidden_dim) # Unused currently
        self.skip_2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip_3 = nn.Linear(hidden_dim, hidden_dim)
        self.skip_4 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Layer 1 Encode edge features and node features
        edge_attr = self.dropout(self.edge_encoder(edge_attr))
        x = self.dropout(self.node_encoder(x))

       

        # Layer 2 GINEConv with edge features
        x_init = x.clone() 
        
        x = self.conv2_a(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn2_a(x) #

        x = self.conv2_b(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn2_b(x) 

       

        # Residual connection for entire GINE block
        x = x + self.skip_2(x_init)  
        x = self.bn2_block(x) 

        x_first_step = x.clone()  

        # Mid layer with TransformerConv    
        x = self.conv_mid2(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn_mid2(x)

        x_init = x.clone() 
        
        x = self.conv3_a(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn3_a(x) 

        x = self.conv3_b(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn3_b(x) 

       
        
        x = x + self.skip_3(x_init)  
        x = self.bn3_block(x) 

        x_second_step = x.clone()  

        # Mid layer with TransformerConv
        x = self.conv_mid(x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.bn_mid(x) 

        
        

        # Accumulated skip connections from previous major blocks
        x = x + x_second_step + x_first_step 
        x = self.bn_mid3(x) 

        # Layer final with skip connection with attention
        x = self.conv_final(x, edge_index, edge_attr=edge_attr)
        x = self.dropout(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn_final(x) 
        
        # FFN + classification
        x = self.pool(x, batch)
        x = self.fc(x)
        return x