import keras
import midi_io
import tqdm

# DIMENSIONS OF INPUT TENSOR
#  1 - number of songs per training batch
#  2 - number of notes per song
#  3 - number of possible notes

event_sequences, scaler = midi_io.load_padded_input_event_sequences()
print(event_sequences.shape)

