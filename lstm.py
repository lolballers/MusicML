import keras as k
import midi_io
import numpy as np

# DIMENSIONS OF INPUT TENSOR
#  1 - number of songs per training batch
#  2 - number of notes per song
#  3 - number of possible notes

# event_sequences, scaler = midi_io.load_padded_input_event_sequences(basename='clementi*format0')
# print(event_sequences)

data = midi_io.load_padded_input_event_sequences(basename='*');

control = []
value = []
velocity = []
time = []

def toOneHot(index):
    onehotvalue = np.zeros(88)
    # if(index < 21 or index > 108):
    #     print(index)
    onehotvalue[int(index - 21)] = 1
    return onehotvalue

def toNote(oneHot):
    total = np.sum(oneHot)
    oneHotNew = np.divide(oneHot, total)
    return np.random.choice(88, p=oneHotNew) + 21

for song in data:
    for note in song:
        control.append(note[0])
        value.append(toOneHot(note[1]))
        velocity.append(note[2])
        time.append(note[3]/100000)

control = np.asarray(control)
value = np.asarray(value)
velocity = np.asarray(velocity)
time = np.asarray(time)

sequenceLength = 20
songLength = 100

controlY = np.roll(control, sequenceLength)
valueY = np.roll(value, sequenceLength)
velocityY = np.roll(velocity, sequenceLength)
timeY = np.roll(time, sequenceLength)

controlmodel = k.models.Sequential();
controlmodel.add(k.layers.Embedding(input_dim = 3, output_dim = 64, input_length=1));
controlmodel.add(k.layers.LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
controlmodel.add(k.layers.Flatten());
controlmodel.add(k.layers.Dense(1, activation="sigmoid"));
controlmodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

controlmodel.fit(control, controlY, epochs=1, verbose=1);
controlstarter = control[0:sequenceLength]
controlsong = [controlstarter];
for ii in range(0,songLength):
    y = controlmodel.predict(controlsong[ii]);
    controlsong.append(np.round(y))

polcontrolsong = []
for array in controlsong:
    for val in array:
        polcontrolsong.append(val)

valuemodel = k.models.Sequential();
valuemodel.add(k.layers.Embedding(input_dim = 2, output_dim = 128, input_length=88));
valuemodel.add(k.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
valuemodel.add(k.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
valuemodel.add(k.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
valuemodel.add(k.layers.Flatten());
valuemodel.add(k.layers.Dense(88, activation="sigmoid"));
valuemodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);

valuemodel.fit(value, valueY, epochs=3, verbose=1);

# valuemodel.save_weights('20valueweights.h5')
valuestarter = value[0:sequenceLength]
valuesong = [valuestarter];
for ii in range(0,songLength):
    y = valuemodel.predict(valuesong[ii]);
    # print(y[0])
    valuesong.append(y)

polvaluesong = []
for array in valuesong:
    for val in array:
        polvaluesong.append(toNote(val))

print(polvaluesong)

velocitymodel = k.models.Sequential();
velocitymodel.add(k.layers.Embedding(input_dim = 1000, output_dim = 128, input_length=1));
velocitymodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
velocitymodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
# velocitymodel.add(k.layers.LSTM(128, activation='tanh', dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
velocitymodel.add(k.layers.Flatten());
velocitymodel.add(k.layers.Dense(1, activation="relu"));
velocitymodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

velocitymodel.fit(velocity, velocityY, epochs=1, verbose=1);

# velocitymodel.save_weights('20velocityweights.h5')
velocitystarter = velocity[0:sequenceLength]
velocitysong = [velocitystarter];
for ii in range(0,songLength):
    y = velocitymodel.predict(velocitysong[ii]);
    velocitysong.append(y)

polvelocitysong = []
for array in velocitysong:
    for val in array:
        polvelocitysong.append(val)

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
timestarter = time[0:sequenceLength]
timesong = [timestarter];
for ii in range(0,songLength):
    y = timemodel.predict(timesong[ii]);
    timesong.append(y)

poltimesong = []
for array in timesong:
    for val in array:
        poltimesong.append(val*100000)

print(poltimesong)
print(len(poltimesong))

song = []

for ii in range(0, len(poltimesong)):
    note = [polcontrolsong[ii], polvaluesong[ii], polvelocitysong[ii], poltimesong[ii]]
    song.append(note)

# realSong = scaler.inverse_transform(song)
#
# for note in realSong:
#     print(note)

midi_io.event_sequence_to_midi(song).save('generated.mid')
# midi_io.event_sequence_to_midi(scaler.inverse_transform(song)).save('generatedtransform.mid')


# model = keras.models.Sequential()
# model.add(keras.layers.LSTM(128, input_shape=(None, 4), stateful=False))
# model.add(keras.layers.Dense(4, activation='relu'))
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# for train_seq in event_sequences[:15]:  # reserve some for validation
#     for curriculum_length in [10, 100]: # start by learning short-term relationships, then progress to big-picture
#         gen = keras.preprocessing.sequence.TimeseriesGenerator(train_seq, train_seq, curriculum_length, batch_size=1)
#         model.fit_generator(gen, epochs=1)





