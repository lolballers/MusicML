import keras
import midi_io
import numpy as np

# DIMENSIONS OF INPUT TENSOR
#  1 - number of songs per training batch
#  2 - number of notes per song
#  3 - number of possible notes

event_sequences, scaler = midi_io.load_padded_input_event_sequences(basename='clementi*format0')
print(event_sequences.shape)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(None, 4), stateful=False))
model.add(keras.layers.Dense(4, activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error')

for train_seq in event_sequences[:15]:  # reserve some for validation
    for curriculum_length in [10, 100]: # start by learning short-term relationships, then progress to big-picture
        gen = keras.preprocessing.sequence.TimeseriesGenerator(train_seq, train_seq, curriculum_length, batch_size=1)
        model.fit_generator(gen, epochs=2)

pass



