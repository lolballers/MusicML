from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist

from mido import MidiFile
from glob import glob

verbosity = 1

# song parsing
songs = []
max_songs = 1
_i = 0
for path in glob('./samples/*.mid'):
    pass


mid = MidiFile('./samples/140.mid')
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)
        if msg.is_meta:
            _i += 1

print("i: {}".format(i))

model = Sequential()

# DIMENSIONS OF INPUT TENSOR
#  1 - number of songs per training batch
#  2 - number of notes per song
#  3 - number of possible notes

