import keras as k
import midi_io
import mido
import numpy as np

# DIMENSIONS OF INPUT TENSOR
#  1 - number of songs per training batch
#  2 - number of notes per song
#  3 - number of possible notes

# event_sequences, scaler = midi_io.load_padded_input_event_sequences(basename='clementi*format0')
# print(event_sequences)

data, scaler = midi_io.load_padded_input_event_sequences(basename='*');
print(data)

control = []
value = []
velocity = []
time = []

for song in data:
    for note in song:
        control.append(note[0])
        value.append(note[1])
        velocity.append(note[2])
        time.append(note[3])

control = np.asarray(control)
value = np.asarray(value)
velocity = np.asarray(velocity)
time = np.asarray(time)

# control, value, velocity, time = midi_io.load_everything()
controlY = np.roll(control, 4)
valueY = np.roll(value, 4)
velocityY = np.roll(velocity, 4)
timeY = np.roll(time, 4)

controlmodel = k.models.Sequential();
controlmodel.add(k.layers.Embedding(input_dim = 2, output_dim = 64, input_length=1));
controlmodel.add(k.layers.LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
controlmodel.add(k.layers.Flatten());
controlmodel.add(k.layers.Dense(1, activation="sigmoid"));
controlmodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

controlmodel.fit(control, controlY, epochs=1, verbose=1);
starter = control[0:4]
controlsong = [starter];
for ii in range(0,100):
    y = controlmodel.predict(controlsong[ii]);
    controlsong.append(np.rint(y))

polcontrolsong = []
for array in controlsong:
    for val in array:
        polcontrolsong.append(val)

print(polcontrolsong)

valuemodel = k.models.Sequential();
valuemodel.add(k.layers.Embedding(input_dim = 1000, output_dim = 128, input_length=1));
valuemodel.add(k.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.8, return_sequences = True));
valuemodel.add(k.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.8, return_sequences = True));
valuemodel.add(k.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.8, return_sequences = True));
valuemodel.add(k.layers.Flatten());
valuemodel.add(k.layers.Dense(1, activation="relu"));
valuemodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

valuemodel.fit(value, valueY, epochs=5, verbose=1);
starter = value[0:4]
valuesong = [starter];
for ii in range(0,100):
    y = valuemodel.predict(valuesong[ii], batch_size=10);
    valuesong.append(y)

polvaluesong = []
for array in valuesong:
    for val in array:
        polvaluesong.append(np.round(val))

print(polvaluesong)

velocitymodel = k.models.Sequential();
velocitymodel.add(k.layers.Embedding(input_dim = 1000, output_dim = 128, input_length=1));
velocitymodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
velocitymodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
# velocitymodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
velocitymodel.add(k.layers.Flatten());
velocitymodel.add(k.layers.Dense(1, activation="relu"));
velocitymodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

velocitymodel.fit(velocity, velocityY, epochs=5, verbose=1);
starter = velocity[0:4]
velocitysong = [starter];
for ii in range(0,100):
    y = velocitymodel.predict(velocitysong[ii]);
    velocitysong.append(y)

polvelocitysong = []
for array in velocitysong:
    for val in array:
        polvelocitysong.append(np.round(val))

print(polvelocitysong)

timemodel = k.models.Sequential();
timemodel.add(k.layers.Embedding(input_dim = 1000, output_dim = 128, input_length=1));
timemodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
timemodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
timemodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
timemodel.add(k.layers.Flatten());
timemodel.add(k.layers.Dense(1, activation="relu"));
timemodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

timemodel.fit(time, timeY, epochs=1, verbose=1);
starter = time[0:4]
timesong = [starter];
for ii in range(0,100):
    y = timemodel.predict(timesong[ii]);
    timesong.append(y)

poltimesong = []
for array in timesong:
    for val in array:
        poltimesong.append(np.round(val))

print(poltimesong)
print(len(poltimesong))

song = []

for ii in range(0, len(poltimesong)):
    note = [polcontrolsong[ii], polvaluesong[ii], polvelocitysong[ii], poltimesong[ii]]
    song.append(note)

print(song)
midi_io.event_sequence_to_midi(song)

# model = keras.models.Sequential()
# model.add(keras.layers.LSTM(128, input_shape=(None, 4), stateful=False))
# model.add(keras.layers.Dense(4, activation='relu'))
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# for train_seq in event_sequences[:15]:  # reserve some for validation
#     for curriculum_length in [10, 100]: # start by learning short-term relationships, then progress to big-picture
#         gen = keras.preprocessing.sequence.TimeseriesGenerator(train_seq, train_seq, curriculum_length, batch_size=1)
#         model.fit_generator(gen, epochs=1)





