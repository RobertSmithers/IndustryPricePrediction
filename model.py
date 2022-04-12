import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The MVP is a simple MLP model. This will be the most basic prediction which can serve as our baseline.
'''
class MVP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MVP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, output_dim)
        )
        
        # Encoders/Embeddings
        # LSTM
        # GraphNN

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.features(x)
        return F.softmax(x, dim=1)