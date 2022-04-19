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
            nn.Linear(input_dim, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1000),
            nn.ReLU()
        )
        
        self.classify = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim),
        )
        
        # Encoders/Embeddings
        # LSTM
        # GraphNN

    def forward(self, x, h_state, c_state):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.features(x)
        x = self.classify(x)
        return F.softmax(x, dim=1)
    
class LSTM_MVP(nn.Module):
    def __init__(self, input_dim, output_dim, lstm_hidden_size=50, lstm_num_layers=3):
        super(LSTM_MVP, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 800),
            nn.ReLU(),
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU()
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(input_size=20, hidden_size=lstm_hidden_size,
                          num_layers=lstm_num_layers, batch_first=True)
        # self.bn = nn.BatchNorm1d(1000, affine=False)  # affine=False would be without a learnable parameter
        
        # self.gru = nn.GRU(input_size=1000, hidden_size=500)
        
        self.decoder = nn.Sequential(
            # nn.Dropout(0.25),      
            nn.Linear(lstm_hidden_size, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, 800),
            nn.ReLU(),
            nn.Linear(800, input_dim)
        )
        
        self.classify = nn.Linear(input_dim, output_dim)
        # Encoders/Embeddings
        # LSTM
        # GraphNN

    # Forward mirrored from https://github.com/Daammon/Stock-Prediction-with-RNN-in-Pytorch/blob/master/Rnn0.ipynb
    def forward(self, x, h):
        batch_size = 1
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.encoder(x)

        x = x.unsqueeze(0)
        output, h = self.lstm(x, h)
        
        output = self.decoder(output)
            
        output = self.classify(output)
        
        return output, h
    
    # https://stackoverflow.com/questions/58781515/lstm-implementation-overfitting
    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        hidden = (weights.new(self.lstm_num_layers, batch_size, self.lstm_hidden_size).zero_().cuda(),
                  weights.new(self.lstm_num_layers, batch_size, self.lstm_hidden_size).zero_().cuda())
        return hidden