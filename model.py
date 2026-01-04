"""Telemanom LSTM Autoencoder Architecture (NASA JPL KDD 2018)"""
import torch
import torch.nn as nn
import config

class TelemanomLSTM(nn.Module):
    """
    LSTM Autoencoder from NASA's Telemanom system
    Paper: "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" (KDD 2018)
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder: Compress sequence to latent representation
        self.encoder = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )
        
        # Bottleneck
        self.encoder_fc = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)
        self.decoder_fc = nn.Linear(config.LATENT_DIM, config.HIDDEN_DIM)
        
        # Decoder: Reconstruct sequence from latent
        self.decoder = nn.LSTM(
            input_size=config.HIDDEN_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )
        
        # Output projection
        self.output = nn.Linear(config.HIDDEN_DIM, config.INPUT_DIM)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Encode
        encoded, (h, c) = self.encoder(x)
        
        # Compress to latent (use last timestep)
        latent = self.encoder_fc(encoded[:, -1, :])
        
        # Expand from latent
        decoder_input = self.decoder_fc(latent)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(decoder_input)
        
        # Reconstruct
        output = self.output(decoded)
        
        return output
    
    def get_reconstruction_error(self, x):
        """Calculate MSE reconstruction error per sample"""
        with torch.no_grad():
            recon = self(x)
            error = ((x - recon) ** 2).mean(dim=(1, 2))  # MSE per sample
        return error