import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SpatialTemporalCSIPredictor:
    def __init__(self, csi_data):
        """
        Initialize the Spatial-Temporal CSI Predictor
        
        Args:
        - csi_data: numpy array of shape (450, 32, 80)
        """
        self.csi_data = csi_data
        self.n_frames, self.n_receivers, self.n_subcarriers = csi_data.shape
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.normalized_data = self.preprocess_data()
        
    def preprocess_data(self):
        """
        Normalize the CSI data using standard scaling
        
        Returns:
        - Normalized CSI data
        """
        # Reshape data for scaling (flatten receivers and subcarriers)
        reshaped_data = self.csi_data.reshape(self.n_frames, -1)
        normalized_data = self.scaler.fit_transform(reshaped_data)
        return normalized_data.reshape(self.n_frames, self.n_receivers, self.n_subcarriers)
    
    def create_sequences(self, sequence_length=10):
        """
        Create input sequences and target values for training
        
        Args:
        - sequence_length: Number of previous frames to use for prediction
        
        Returns:
        - X: Input sequences
        - y: Target values
        """
        X, y = [], []
        for i in range(len(self.normalized_data) - sequence_length):
            X.append(self.normalized_data[i:i+sequence_length])
            y.append(self.normalized_data[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    class SpatialTemporalLSTM(nn.Module):
        def __init__(self, n_receivers, n_subcarriers, hidden_size, num_layers):
            super().__init__()
            self.n_receivers = n_receivers
            self.n_subcarriers = n_subcarriers
            
            # LSTM layer that processes each receiver's subcarrier data
            self.lstm = nn.LSTM(
                input_size=n_subcarriers,  # Process each receiver's subcarriers 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True
            )
            
            # Spatial attention mechanism
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=hidden_size, 
                num_heads=4
            )
            
            # Output layer to reconstruct the full CSI matrix
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, n_subcarriers),
                nn.ReLU()
            )
        
        def forward(self, x):
            # x shape: (batch_size, seq_length, receivers, subcarriers)
            batch_size, seq_len, receivers, subcarriers = x.shape
            
            # Process each receiver's subcarrier data through LSTM
            lstm_outputs = []
            for r in range(receivers):
                # Extract data for this receiver
                receiver_data = x[:, :, r, :]  # (batch_size, seq_len, subcarriers)
                
                # LSTM processing for this receiver
                receiver_lstm_out, _ = self.lstm(receiver_data)
                
                # Take the last time step output
                lstm_outputs.append(receiver_lstm_out[:, -1, :])
            
            # Stack receiver outputs
            lstm_outputs = torch.stack(lstm_outputs, dim=1)  # (batch_size, receivers, hidden_size)
            
            # Spatial attention
            attention_out, _ = self.spatial_attention(
                lstm_outputs, lstm_outputs, lstm_outputs
            )
            
            # Reconstruct subcarrier data for each receiver
            prediction = self.fc(attention_out).view(
                batch_size, receivers, subcarriers
            )
            
            return prediction
    
    def train_model(self, sequence_length=10, epochs=100, batch_size=32):
        """
        Train the spatial-temporal LSTM model
        
        Args:
        - sequence_length: Number of previous frames to use for prediction
        - epochs: Number of training epochs
        - batch_size: Training batch size
        
        Returns:
        - Trained model
        """
        # Create sequences
        X, y = self.create_sequences(sequence_length)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
        # Initialize model
        model = self.SpatialTemporalLSTM(
            n_receivers=self.n_receivers, 
            n_subcarriers=self.n_subcarriers, 
            hidden_size=64, 
            num_layers=2
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        return model
    
    def predict_next_frame(self, model, last_sequences):
        """
        Predict the next frame using the trained model
        
        Args:
        - model: Trained spatial-temporal model
        - last_sequences: Last sequence of frames
        
        Returns:
        - Predicted next frame (normalized)
        """
        model.eval()
        with torch.no_grad():
            prediction = model(torch.FloatTensor(last_sequences))
        
        return prediction.numpy()
    
    def inverse_transform(self, normalized_prediction):
        """
        Convert normalized prediction back to original scale
        
        Args:
        - normalized_prediction: Prediction in normalized space
        
        Returns:
        - Prediction in original scale
        """
        # Reshape for inverse transform
        reshaped_pred = normalized_prediction.reshape(-1, self.n_receivers * self.n_subcarriers)
        original_scale_pred = self.scaler.inverse_transform(reshaped_pred)
        return original_scale_pred.reshape(self.n_receivers, self.n_subcarriers)

def main():
    # Load data
    with open(r"C:\Users\nrazavi\Downloads\batch1_1Lane_450U_1Rx_32Tx_1T_80K_2620000000.0fc.pickle", 'rb') as f:
        data = pickle.load(f)

    # Directly use freq_channel (450 x 32 x 80 matrix)
    freq_channel = np.squeeze(data["freq_channel"])
    
    # Verify dimensions
    print("CSI Data Shape:", freq_channel.shape)
    
    # Initialize predictor
    predictor = SpatialTemporalCSIPredictor(np.real(freq_channel[:448,:,:]))
    
    # Train model
    trained_model = predictor.train_model()
    
    # Prepare last sequences for prediction
    last_sequences = predictor.normalized_data[-10:]
    last_sequences = last_sequences.reshape(1, 10, predictor.n_receivers, predictor.n_subcarriers)
    
    # Predict next frame
    predicted_frame_normalized = predictor.predict_next_frame(
        trained_model, last_sequences
    )
    
    # Convert back to original scale
    predicted_frame = predictor.inverse_transform(predicted_frame_normalized)
    
    print("Predicted Next Frame Shape:", predicted_frame.shape)
    print("Predicted Next Frame Preview:\n", predicted_frame)
    print("Actual Next Frame Preview:\n", np.real(freq_channel[449,:,:]))

if __name__ == "__main__":
    main()