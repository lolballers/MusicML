import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import np_utils
from keras.optimizers import RMSprop
import music21
import glob

# Constants
sequence_len = 30
hidden_dim = 500
b_size = 128
dropout = .2
song_len = 500
epochs_until_song = 15

# List to store all the raw music files
music_files = []

# Read in all the files that music21 has
for file in glob.glob("C:/Program Files/Python36/Lib/site-packages/music21/corpus/bach/*.mxl"):
    music_files.append(music21.converter.parse(file))

for file in glob.glob("C:/Program Files/Python36/Lib/site-packages/music21/corpus/mozart/k155/*.mxl"):
    music_files.append(music21.converter.parse(file))

for file in glob.glob("C:/Program Files/Python36/Lib/site-packages/music21/corpus/beethoven/*.mxl"):
    music_files.append(music21.converter.parse(file))

# List to store all the note pitches (each as a string)
notes = []

# Go through every music file, remove rhythm and put notes into "notes"
for song in music_files:
    song_notes = None

    # Check what parts are in the current song
    parts = music21.instrument.partitionByInstrument(song)

    if parts:
        # If there's more than one part, choose the first one
        song_notes = parts.parts[0].recurse()
    else:
        # Otherwise just take all the notes
        song_notes = song.flat.notes

    # Now go through every element
    for element in song_notes:
        # Check whether the current element is a note
        if isinstance(element, music21.note.Note):
            if element.pitch != '':
                notes.append(str(element.pitch))
        # Check whether the current element is a chord
        elif isinstance(element, music21.chord.Chord):
            # Save chords with their note values separated by '.'
            notes.append('.'.join(str(n) for n in element.normalOrder))

# A set of all the distinct names for pitches
pitch_names = sorted(set(item for item in notes))
# The actual number of notes, used for creating one-hot vectors
distinct_notes = len(pitch_names)
# Two dictionaries for going between string pitch names and integer pitch indices
note_to_int = {note: index for index, note in enumerate(pitch_names)}
int_to_note = {index: note for index, note in enumerate(pitch_names)}

# Two numpy arrays, one for the sequence_len input sequences and one for the single value expected
# output for each of those sequences. Each of note value is stored as its one-hot representation
input_seq = np.zeros((len(notes) - sequence_len, sequence_len, distinct_notes))
output_seq = np.zeros((len(notes) - sequence_len, distinct_notes))

# Go through every element and create those input/output sequence pairs
for i in range(0, (len(notes) - sequence_len)):
    input_seq[i] = [np_utils.to_categorical(note_to_int[note], distinct_notes) for note in notes[i:i+sequence_len]]
    output_seq[i] = np_utils.to_categorical(note_to_int[notes[i + sequence_len]], distinct_notes)

# Create the 3-layer LSTM + Dense layer, given the specified dropout and hidden_dim values
model = Sequential()
model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(sequence_len, distinct_notes), dropout=dropout))
model.add(LSTM(hidden_dim, return_sequences=True, dropout=dropout))
model.add(LSTM(hidden_dim, dropout=dropout))
model.add(Dense(distinct_notes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

# Used for saving the midi file
cnt = 1
# Used for making rhythm
offset = 0

while True:
    # Fit the model for epochs_until_song epochs
    model.fit(input_seq, output_seq, batch_size=b_size, epochs=epochs_until_song)

    # Choose a random starting sequence
    start_ind = np.random.randint(len(notes) - sequence_len)
    seed = input_seq[start_ind]
    pred_out = []

    # Then for each of the next notes that we want to generated
    for curr_index in range(song_len):
        # Reshape the seed input into something that can go into the network
        prediction_input = np.reshape(seed, (1, sequence_len, distinct_notes))

        # Predict the next note given the input seed
        prediction = model.predict(prediction_input, verbose=0)

        # Take the max value from the output one_hot vector and append it to the input seed
        index = np.argmax(prediction)
        pred_out.append(index)

        # Moves the input seed over one to the right
        temp_pattern = np.zeros(pattern.shape)
        temp_pattern[0:sequence_len - 1, :] = pattern[1:, :]
        temp_pattern[sequence_len - 1] = prediction
        pattern = temp_pattern

    output_notes = []

    # Go through all the generate outputs and convert them into Music21 Notes and Chords
    for index in range(len(pred_out)):
        # Because we stored chords strings of pitches with a '.', this checks whether a note is a chord
        if int_to_note[pred_out[index]].find(".") != -1:
            chord_notes = int_to_note[pred_out[index]].split('.')
            temp_notes = []
            for current_note in chord_notes:
                new_note = music21.note.Note(int(current_note))
                temp_notes.append(new_note)
            new_chord = music21.chord.Chord(temp_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # If the note doesn't have a '.' then its a note
        else:
            new_note = music21.note.Note(int_to_note[pred_out[index]])
            new_note.offset = offset
            output_notes.append(new_note)
        # Everything has offset .5
        offset += .5

    # Convert the list of notes and chords into a Music21 Stream and write it as a .midi file
    midi_stream = music21.stream.Stream(output_notes)
    midi_stream.write("midi", fp="music_output_{}.mid".format(cnt))
    cnt += 1