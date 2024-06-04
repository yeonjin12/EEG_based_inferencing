#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
## Version history:

2018:
    Original script by Dr. Luis Manso [lmanso], Aston University
    
2019, June:
    Revised, commented and updated by Dr. Felipe Campelo [fcampelo], Aston University
    (f.campelo@aston.ac.uk / fcampelo@gmail.com)
"""

import numpy as np
from pylsl import StreamInlet, resolve_stream
from EEG_feature_extraction import generate_feature_vectors_from_samples_matrix
import time

# num: 측정 횟수
# period: 특성벡터 뽑는 시간 간격

global period
period = 1

def process_real_time_data(real_time_data, cols_to_ignore, real_time_output):
    """
    Processes real-time data to extract features and append to the existing output array.
    
    Parameters:
        real_time_data (np.ndarray): 2D numpy array with columns [timestamps, TP9, AF7, AF8, TP10, Right AUX]
        cols_to_ignore (list): list of columns to ignore from the data
        real_time_output (np.ndarray): 2D array to store the extracted features
    
    Returns:
        np.ndarray: Updated 2D array containing the extracted features
    """
    # Set the state manually or based on some criteria
    state = 1.0  # Assuming 'neutral' for example // inferencing할 땐 마지막 열 제외해야함 - TODO
    
    # Generate feature vectors from the real-time data
    vectors, header = generate_feature_vectors_from_samples_matrix(
        matrix=real_time_data,
        nsamples=150,
        period=period,
        state=state,
        remove_redundant=True,
        cols_to_ignore=cols_to_ignore
    )

    print("real_time_data : ", real_time_data.shape)
    print('Resulting vector shape for the data:', vectors.shape)
    
    if real_time_output is None:
        real_time_output = vectors
    else:
        real_time_output = np.vstack([real_time_output, vectors])
    
    return real_time_output

import torch
import torch.nn as nn
import numpy as np

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # (batch_size, input_dim) -> (batch_size, 512)
        x = x.unsqueeze(1)  # (batch_size, 512) -> (batch_size, 1, 512)
        x = x.permute(1, 0, 2)  # (batch_size, 1, 512) -> (1, batch_size, 512)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # (1, batch_size, 512) -> (batch_size, 512)
        x = self.fc(x)  # (batch_size, 512) -> (batch_size, num_classes)
        return x

class MentalStatePredictor:
    def __init__(self, model_path, input_dim, num_classes):
        self.model = TransformerClassifier(input_dim, num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.input_dim = input_dim

    def predict(self, input_vector):
        assert len(input_vector) == self.input_dim, f"Input vector must be of size {self.input_dim}"
        with torch.no_grad():
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)  # (1, input_dim)
            output = self.model(input_tensor)
            _, predicted_label = torch.max(output, 1)
            return predicted_label.item()

def main():
    #Transformer loading
    model_path = 'transformer_classifier_model.pt'
    input_dim = 988
    num_classes = 3

    predictor = MentalStatePredictor(model_path, input_dim, num_classes)
    input_vector = np.random.rand(988)  # Example input vector of size [988]

    # Resolve an EEG stream on the lab network
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # Create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    # Channels to be used
    cols_to_ignore = [0]  # Assuming we ignore the timestamp column

    # Initialize the real_time_output array
    real_time_output = None

    print("Start receiving data...")
    eeg_num = 0

    while True:
        # Get EEG data sample (timestamp, [TP9, AF7, AF8, TP10, Right AUX])
        sample, timestamp = inlet.pull_sample()
        sample = [timestamp] + sample
        # print(sample)

        # Append the new sample to a buffer (150 samples for example)
        if 'data_buffer' not in locals():
            data_buffer = []
        data_buffer.append(sample)
        # print("data_buffer:", data_buffer )
        # If buffer is full (150 samples), process the data
        if len(data_buffer) > 1000:
            headers = ['timestamps', 'TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
            real_time_data = np.array(data_buffer)
            inlet_time = real_time_data[-1, 0] - real_time_data[0, 0]
            if inlet_time > eeg_num * 0.5 * period:
                real_time_output = process_real_time_data(real_time_data, cols_to_ignore, real_time_output)
                eeg_num += 1

                input_vector = real_time_output[-1, :-1]
                predicted_label = predictor.predict(input_vector)
                print(f"Predicted Label: {predicted_label}")


            # Clear the buffer after processing
            # data_buffer = []
            # Print the updated real_time_output (optional)
            #print("real_time_data : ", real_time_data)
            #print("inlet time : ", inlet_time)


if __name__ == '__main__':
    main()