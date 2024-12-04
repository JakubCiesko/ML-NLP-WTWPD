import torch 
import torch.nn as nn 


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim=300, hidden_dim=2048, dropout_rate=0.1, smoothing_coeff=0.2, leaky_relu_slope=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout_rate),         
            nn.Linear(input_dim, hidden_dim), 
            nn.LeakyReLU(leaky_relu_slope),             
            nn.Linear(hidden_dim, hidden_dim),  
            nn.LeakyReLU(leaky_relu_slope), 
            nn.Linear(hidden_dim, hidden_dim),  
            nn.LeakyReLU(leaky_relu_slope),                 
            #nn.Dropout(dropout_rate),           
            nn.Linear(hidden_dim, 1)         
        )
        self.smoothing_coeff = smoothing_coeff
    
    def forward(self, x):
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        return probs 
