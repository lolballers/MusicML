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

print(time)

sequenceLength = 500
songLength = 4

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

controlmodel.fit(controlY[sequenceLength:], control[sequenceLength:], epochs=20, verbose=1, batch_size=128);
controlstarter = control[0:sequenceLength]
controlsong = [controlstarter];
for ii in range(0,songLength):
    y = controlmodel.predict(controlsong[ii]);
    controlsong.append(np.round(y))

polcontrolsong = []
for array in controlsong:
    for val in array:
        polcontrolsong.append(val)

#this LSTM takes up so much time
valuemodel = k.models.Sequential();
valuemodel.add(k.layers.Embedding(input_dim = 2, output_dim = 128, input_length=88));
valuemodel.add(k.layers.LSTM(128, activation= "relu", dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
valuemodel.add(k.layers.LSTM(128, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
valuemodel.add(k.layers.LSTM(128, activation= "relu", dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
valuemodel.add(k.layers.Flatten());
valuemodel.add(k.layers.Dense(88, activation="softmax"));
valuemodel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']);

valuemodel.fit(valueY[sequenceLength:], value[sequenceLength:], epochs=20, verbose=1);

# valuemodel.save_weights('20valueweightsrac.h5')
valuestarter = value[0:sequenceLength]
valuesong = [valuestarter];
for ii in range(0,songLength):
    y = valuemodel.predict(valuesong[ii]);
    valuesong.append(y)

polvaluesong = []
for array in valuesong:
    for val in array:
        polvaluesong.append(toNote(val))

print(polvaluesong)

oneVelocity = []
valueVelocity = []

for velocityVal in velocity:
    if(velocityVal!=0.0):
        oneVelocity.append(1)
        valueVelocity.append(velocityVal)
    else:
        oneVelocity.append(velocityVal)

oneVelocityY = np.roll(oneVelocity, sequenceLength)
valueVelocityY = np.roll(valueVelocity, sequenceLength)

oneVelocitymodel = k.models.Sequential();
oneVelocitymodel.add(k.layers.Embedding(input_dim = 1000, output_dim = 128, input_length=1));
oneVelocitymodel.add(k.layers.LSTM(128, activation= "sigmoid", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
oneVelocitymodel.add(k.layers.LSTM(128, activation= "sigmoid", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
oneVelocitymodel.add(k.layers.LSTM(128, activation="sigmoid", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
oneVelocitymodel.add(k.layers.Flatten());
oneVelocitymodel.add(k.layers.Dense(1, activation="tanh"));
oneVelocitymodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

oneVelocitymodel.fit(np.array(oneVelocityY[sequenceLength:]), np.array(oneVelocity[sequenceLength:]), epochs=20, verbose=1);
oneVelocitystarter = oneVelocity[0:sequenceLength]
oneVelocitysong = [oneVelocitystarter];

for ii in range(0,songLength):
    y = oneVelocitymodel.predict(oneVelocitysong[ii]);
    choices = []
    for array in y:
        for val in array:
            choices.append(np.random.choice(2, p=[1-val, val]))
    oneVelocitysong.append(choices)

print(oneVelocitysong)

velocitymodel = k.models.Sequential();
velocitymodel.add(k.layers.Embedding(input_dim = 1000, output_dim = 128, input_length=1));
velocitymodel.add(k.layers.LSTM(128, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
velocitymodel.add(k.layers.LSTM(128, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
velocitymodel.add(k.layers.LSTM(128, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
velocitymodel.add(k.layers.Flatten());
velocitymodel.add(k.layers.Dense(1, activation="relu"));
velocitymodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

velocitymodel.fit(np.array(valueVelocityY[sequenceLength:]), np.array(valueVelocity[sequenceLength:]), epochs=20, verbose=1);
valueVelocitystarter = valueVelocity[0:sequenceLength]
valueVelocitysong = [valueVelocitystarter];
for ii in range(0,songLength):
    y = velocitymodel.predict(np.array(valueVelocitysong[ii]));
    valueVelocitysong.append(y)

print(valueVelocitysong)

polValueVelocitySong = []

for array in valueVelocitysong:
    for val in array:
        polValueVelocitySong.append(val)

polvelocitysong = []
j = sequenceLength
for array in oneVelocitysong:
    for val in array:
        add = 0;
        if(val!=0):
            add = polValueVelocitySong[j]
            j = j+1
        polvelocitysong.append(np.round(add))

print(polvelocitysong)

# velocitymodel.fit(velocityY[sequenceLength:], velocity[sequenceLength:], epochs=1, verbose=1);

# velocitymodel.save_weights('20velocityweights.h5')
# velocitystarter = velocity[0:sequenceLength]
# velocitysong = [velocitystarter];
# for ii in range(0,songLength):
#     y = velocitymodel.predict(velocitysong[ii]);
#     velocitysong.append(y)
#
# polvelocitysong = []
# for array in velocitysong:
#     for val in array:
#         polvelocitysong.append(np.round(val))
#
# print(polvelocitysong)

oneTimemodel = k.models.Sequential();
oneTimemodel.add(k.layers.Embedding(input_dim = 1000, output_dim = 128, input_length=1));
oneTimemodel.add(k.layers.LSTM(128, activation= "sigmoid", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
oneTimemodel.add(k.layers.LSTM(128, activation= "sigmoid", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
oneTimemodel.add(k.layers.LSTM(128, activation="sigmoid", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
oneTimemodel.add(k.layers.Flatten());
oneTimemodel.add(k.layers.Dense(1, activation="tanh"));
oneTimemodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

oneTime = []
valueTime = []

for timeVal in time:
    if(timeVal!=0.0):
        oneTime.append(1)
        valueTime.append(timeVal)
    else:
        oneTime.append(timeVal)

oneTimeY = np.roll(oneTime, sequenceLength)
valueTimeY = np.roll(valueTime, sequenceLength)

oneTimemodel.fit(oneTimeY[sequenceLength:], oneTime[sequenceLength:], epochs=20, verbose=1);
oneTimestarter = oneTime[0:sequenceLength]
oneTimesong = [oneTimestarter];

for ii in range(0,songLength):
    y = oneTimemodel.predict(oneTimesong[ii]);
    choices = []
    for array in y:
        for val in array:
            choices.append(np.random.choice(2, p=[1-val, val]))
    oneTimesong.append(choices)

print(oneTimesong)

timemodel = k.models.Sequential();
timemodel.add(k.layers.Embedding(input_dim = 1000, output_dim = 128, input_length=1));
timemodel.add(k.layers.LSTM(128, activation= "relu", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
timemodel.add(k.layers.LSTM(128, activation= "relu", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
timemodel.add(k.layers.LSTM(128, activation="relu", dropout=0.1, recurrent_dropout=0.1, return_sequences = True));
timemodel.add(k.layers.Flatten());
timemodel.add(k.layers.Dense(1, activation="relu"));
timemodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

timemodel.fit(valueTimeY[sequenceLength:], valueTime[sequenceLength:], epochs=20, verbose=1);
valueTimestarter = valueTime[0:sequenceLength]
valueTimesong = [valueTimestarter];
for ii in range(0,songLength):
    y = timemodel.predict(valueTimesong[ii]);
    valueTimesong.append(y)

print(valueTimesong)

polValueTimeSong = []

for array in valueTimesong:
    for val in array:
        polValueTimeSong.append(val)

poltimesong = []
i = sequenceLength
for array in oneTimesong:
    for val in array:
        add = 0;
        if(val!=0):
            add = polValueTimeSong[i]
            i = i+1
        poltimesong.append(add*100000)

print(poltimesong)

song = []

for ii in range(0, len(poltimesong)):
    note = [polcontrolsong[ii], polvaluesong[ii], polvelocitysong[ii], poltimesong[ii]]
    song.append(note)

# realSong = scaler.inverse_transform(song)
#
# for note in realSong:
#     print(note)

midi_io.event_sequence_to_midi(song).save('generated20.mid')
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





