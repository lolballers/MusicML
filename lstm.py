import keras as k
import midi_io
import numpy as np
#for plotting stuff
import matplotlib.pyplot as plt

data = midi_io.load_padded_input_event_sequences(basename='*');

#notes are composed of four components:
#   control - value 1 or 2, whether the note is a note or a control (we don't care about controls, it's just notes)
#   value - value 21 - 103, note pitch
#   velocity - how loud the note is
#   time - rhythm of the note

control = []
value = []
velocity = []
time = []

numMidiNotes = 88;

#creates one hot vectors given an index for the one
def toOneHot(index):
    onehotvalue = np.zeros(numMidiNotes)
    #midi goes between 22 - 103 not 1 - 88
    onehotvalue[int(index - 21)] = 1
    return onehotvalue

def toNote(oneHot):
    total = np.sum(oneHot)
    #makes everything add up to 1 so it's a vaild probability distribution
    oneHotNew = np.divide(oneHot, total)
    #midi goes between 22 - 103 not 1 - 88
    #sampling based on probability distribution
    return np.random.choice(numMidiNotes, p=oneHotNew) + 21

#the time value is ludicrously large and needs to be divided by this factor
timeConversionFactor = 100000

#separating notes into their four components so that we can train each in separate models
for song in data:
    for note in song:
        control.append(note[0])
        value.append(toOneHot(note[1]))
        velocity.append(note[2])
        time.append(note[3]/ timeConversionFactor)

control = np.asarray(control)
value = np.asarray(value)
velocity = np.asarray(velocity)
time = np.asarray(time)

#the sequence length to train on
sequenceLength = 500
#how many times to produce the given sequence (the song will be sequenceLength * songLength notes long)
songLength = 10

#Y is a bit misleading as it's actually the X value. But it's the transformed version which is why it's named Y
#Rolling an array by 1 is like turning (1,2,3,4,5) into (5,1,2,3,4)
controlY = np.roll(control, sequenceLength)
valueY = np.roll(value, sequenceLength)
velocityY = np.roll(velocity, sequenceLength)
timeY = np.roll(time, sequenceLength)

#method to create LSTM models
def createModel(inputDim, inputLength, activation, denseActivation, numLayers, loss):
    model = k.models.Sequential();
    #Embedding layer turns input into 3D array which can then be passed into LSTM
    #output dim says that the numbers in the output of the embedding layer will be between 1 and output_dim
    model.add(k.layers.Embedding(input_dim = inputDim, output_dim = 64, input_length=inputLength));
    for ii in range(1, numLayers):
        model.add(k.layers.LSTM(128, activation=activation, dropout=0.2, recurrent_dropout=0.2, return_sequences = True));
    #Flatten is necessary to turn 3D array back into 1D to pass through Dense layer
    model.add(k.layers.Flatten());
    model.add(k.layers.Dense(inputLength, activation=denseActivation));
    model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy']);
    return model;

#it looks like a lot of magic numbers, but these are hyperparameters but have been tuned a lot of times
controlmodel = createModel(3, 1, 'tanh', 'sigmoid', 1, 'mean_squared_error')

#model to predict the control
#we rolled by sequence length; imagine we rolled (1,2,3,4,5) by 1 so we have (5,1,2,3,4)
#if splice each array ignoring the first sequence length values we get
#(2,3,4,5) and (1,2,3,4): perfect
controlmodel.fit(controlY[sequenceLength:], control[sequenceLength:], epochs=3, verbose=1, batch_size=128);

#calls model.predict for songlength to produce our song
def modelPredict(model, song, round):
    for ii in range(0, songLength):
        y = model.predict(song[ii])
        if(round):
            song.append(np.round(y))
        else:
            song.append(y)

#output is weirdly formatted in random arrays, this just peels back everything to make it a 1D array
def unwrap(song):
    polsong = []
    for array in song:
        for val in array:
            polsong.append(val)
    return polsong;

#take the first sequencelength notes to start as a seed
controlstarter = control[0:sequenceLength]
controlsong = [controlstarter];

modelPredict(controlmodel, controlsong, True)

polcontrolsong = unwrap(controlsong)
print(polcontrolsong)

#this LSTM takes up so much time because the inputlength is 88
valuemodel = createModel(2, numMidiNotes, 'relu', 'softmax', 3, 'categorical_crossentropy');

valuemodel.fit(valueY[sequenceLength:], value[sequenceLength:], epochs=1, verbose=1);

# it's the only LSTM worth saving because it takes so darn long to train
# valuemodel.save_weights('20valueweightsrac.h5')

valuestarter = value[0:sequenceLength]
valuesong = [valuestarter];


modelPredict(valuemodel, valuesong, False)

polvaluesong = []
for array in valuesong:
    for val in array:
        polvaluesong.append(toNote(val))

print(polvaluesong)


#we're going to use a strange approach for the velocity and time LSTM models
#the output needs to consist of large numbers and zeros
#unfortunately this is hard to do with one model
#so what we are going to do is first produce a model that outputs a sequence of 1s and 0s
#then we'll produce a sequence of large numbers
#and then jam the two together, replacing 1s in the 1s and 0s sequence with large numbers
#it's not pretty but it works

#the 1s and 0s list
oneVelocity = []
#the large numbers list
valueVelocity = []

#splitting original velocity list into oneVelocity and valueVelocity
for velocityVal in velocity:
    if(velocityVal!=0.0):
        oneVelocity.append(1)
        valueVelocity.append(velocityVal)
    else:
        oneVelocity.append(velocityVal)

#rolling as explained previously
oneVelocityY = np.roll(oneVelocity, sequenceLength)
valueVelocityY = np.roll(valueVelocity, sequenceLength)

oneVelocitymodel = createModel(1000, 1, 'sigmoid', 'tanh', 3, 'mean_squared_error')

oneVelocitymodel.fit(np.array(oneVelocityY[sequenceLength:]), np.array(oneVelocity[sequenceLength:]), epochs=10, verbose=1);
oneVelocitystarter = oneVelocity[0:sequenceLength]
oneVelocitysong = [oneVelocitystarter];

for ii in range(0,songLength):
    y = oneVelocitymodel.predict(oneVelocitysong[ii]);
    choices = []
    #unwrapping stuff because of the weird formatting of the output of the neural net
    for array in y:
        for val in array:
            #chooses a 1 or 0 based on probability given by predicted val
            choices.append(np.random.choice(2, p=[1-val, val]))
    oneVelocitysong.append(choices)


velocitymodel = createModel(1000, 1, 'relu', 'relu', 3, 'mean_squared_error')

velocitymodel.fit(np.array(valueVelocityY[sequenceLength:]), np.array(valueVelocity[sequenceLength:]), epochs=10, verbose=1);
valueVelocitystarter = valueVelocity[0:sequenceLength]
valueVelocitysong = [valueVelocitystarter];

for ii in range(0,songLength):
    y = velocitymodel.predict(np.array(valueVelocitysong[ii]));
    valueVelocitysong.append(y)


polValueVelocitySong = []

for array in valueVelocitysong:
    for val in array:
        polValueVelocitySong.append(val)

polvelocitysong = []
j = sequenceLength

#putting in large numbers from the large numbers list whenever there is a zero in the oneVelocitysong
for array in oneVelocitysong:
    for val in array:
        add = 0;
        if(val!=0):
            add = polValueVelocitySong[j]
            j = j+1
        polvelocitysong.append(np.round(add))

print(polvelocitysong)

#same idea down here as the oneVelocitymodel, not going to repeat comments

oneTimemodel = createModel(1000, 1, 'sigmoid', 'tanh', 3, 'mean_squared_error')

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

oneTimemodel.fit(oneTimeY[sequenceLength:], oneTime[sequenceLength:], epochs=10, verbose=1);
oneTimestarter = oneTime[0:sequenceLength]
oneTimesong = [oneTimestarter];

for ii in range(0,songLength):
    y = oneTimemodel.predict(oneTimesong[ii]);
    choices = []
    for array in y:
        for val in array:
            choices.append(np.random.choice(2, p=[1-val, val]))
    oneTimesong.append(choices)


timemodel = createModel(1000, 1, 'relu', 'relu', 3, 'mean_squared_error')

timemodel.fit(valueTimeY[sequenceLength:], valueTime[sequenceLength:], epochs=10, verbose=1);
valueTimestarter = valueTime[0:sequenceLength]
valueTimesong = [valueTimestarter];

modelPredict(timemodel, valueTimesong, False)


polValueTimeSong = unwrap(valueTimesong)

poltimesong = []
i = 0
for array in oneTimesong:
    for val in array:
        add = 0;
        if(val!=0):
            add = polValueTimeSong[i]
            i = i+1
        #remember that there was that conversion factor up at the top because the numbers were too large
        poltimesong.append(add * timeConversionFactor)

print(poltimesong)

song = []

#Final step yay! Now we just jam all four components back together to produce notes.
for ii in range(0, len(poltimesong)):
    note = [polcontrolsong[ii], polvaluesong[ii], polvelocitysong[ii], poltimesong[ii]]
    song.append(note)

#thanks Jens for this handy function, now we'll be able to listen to our modern classical music in a file called generated.mid
midi_io.event_sequence_to_midi(song).save('generated.mid')






