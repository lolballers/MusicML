import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.utils import np_utils
from keras.optimizers import RMSprop
import music21
import glob

# Constants
sequence_len = 30
hidden_dim = 256
b_size = 128
note_offset = 0.5

music_files = []

for file in glob.glob("C:/Users/peter/PycharmProjects/MLTest/venv/Lib/site-packages/music21/corpus/bach/*.mxl"):
    music_files.append(music21.converter.parse(file))

notes = []

for song in music_files:
    song_notes = None

    parts = music21.instrument.partitionByInstrument(song)

    if parts:
        song_notes = parts.parts[0].recurse()
    else:
        song_notes = song.flat.notes

    for element in song_notes:
        if isinstance(element, music21.note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, music21.chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print(notes)

pitch_names = sorted(set(item for item in notes))
note_to_int = {note: index for index, note in enumerate(pitch_names)}
int_to_note = {index: note for index, note in enumerate(pitch_names)}
distinct_notes = len(pitch_names)

print(pitch_names)
print(note_to_int)
print(int_to_note)
print(distinct_notes)

input_seq = np.zeros((len(notes) - sequence_len, sequence_len, distinct_notes))
output_seq = np.zeros((len(notes) - sequence_len, distinct_notes))

for i in range(0, (len(notes) - sequence_len)):
    input_seq[i] = [np_utils.to_categorical(note_to_int[note], distinct_notes) for note in notes[i:i+sequence_len]]
    output_seq[i] = np_utils.to_categorical(note_to_int[notes[i + sequence_len]], distinct_notes)

n_patterns = len(input_seq)

model = Sequential()
model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(sequence_len, distinct_notes)))
model.add(LSTM(hidden_dim, return_sequences=True))
model.add(LSTM(hidden_dim))
model.add(Dense(distinct_notes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

cnt = 1

while True:
    model.fit(input_seq, output_seq, batch_size=b_size, epochs=5)

    start_ind = np.random.randint(len(notes) - sequence_len)
    pattern = input_seq[start_ind]
    pred_out = []

    for curr_index in range(500):
        prediction_input = np.reshape(pattern, (1, sequence_len, distinct_notes))

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        pred_out.append(result)

        temp_pattern = np.zeros(pattern.shape)
        temp_pattern[0:sequence_len - 1, :] = pattern[1:, :]
        temp_pattern[sequence_len - 1] = np_utils.to_categorical(index, distinct_notes)
        pattern = temp_pattern

    offset = 0
    output_notes = []

    for note in pred_out:
        if('.' in pattern) or note.isdigit():
            chord_notes = note.split('.')
            temp_notes = []
            for current_note in chord_notes:
                new_note = music21.note.Note(int(current_note))
                temp_notes.append(new_note)
            new_chord = music21.chord.Chord(temp_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = music21.note.Note(note)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += .5

    midi_stream = music21.stream.Stream(output_notes)
    midi_stream.write("midi", fp="music_output_{}.mid".format(cnt))
    cnt += 1