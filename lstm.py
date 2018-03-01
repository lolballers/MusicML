import mido
import numpy as np
import keras

# NOTES ABOUT FORMATTING
#
# Important Controls:
# 7 = Volume
# 10 = Pan
# 91 = Reverb
# 64 = Sustain


_num_notes = 128 # in midi standard
_dt = 0.001 # seconds/frame
_dtype = np.float32

def midi_to_event_sequence(mid: mido.MidiFile) -> np.ndarray:
    if mid.type > 1:
        raise NotImplementedError('parser only handles type 0-1 files')

    tempo = 500000 # default according to MIDI standard
    all_events = np.concatenate(tuple(mid.tracks))
    all_events = [event for event in all_events if event.type == 'set_tempo' or event.type == 'note_on' or event.type == 'control_change']
    sequence_length = sum([1 for event in all_events if event.type == 'note_on' or event.type == 'control_change'])
    event_sequence = np.zeros((sequence_length, 4), dtype=_dtype)

    dt = 0
    i = 0
    for event in all_events:
        dt += tempo * event.time / mid.ticks_per_beat
        if event.type == 'set_tempo':
            tempo = event.tempo
        elif event.type == 'note_on' or event.type == 'control_change':

            if event.type == 'note_on':
                event_sequence[i] = np.array((1, event.note, event.velocity, dt), dtype=_dtype)
            elif event.type == 'control_change':
                event_sequence[i] = np.array((2, event.control, event.value, dt), dtype=_dtype)

            i += 1
            dt = 0

    return event_sequence


def event_sequence_to_midi(event_sequence: np.ndarray) -> mido.MidiFile:
    ticks_per_beat = 480
    tempo = 500000

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(mido.Message('program_change', channel=0, program=0, time=0))
    for i, event in enumerate(event_sequence):
        if event[0] == 1: # note_on
            track.append(mido.Message('note_on',
                                      note = int(event[1]),
                                      velocity = int(event[2]),
                                      time = int(event[3] * ticks_per_beat / tempo)
                                      ))
        if event[0] == 2: # control_change
            track.append(mido.Message('control_change',
                                      control = int(event[1]),
                                      value = int(event[2]),
                                      time = int(event[3] * ticks_per_beat / tempo)
                                      ))
    return mid


def test_event_sequence_midi():
    mid = mido.MidiFile('./samples/midi/elise_format0.mid')
    event_sequence_to_midi(midi_to_event_sequence(mid)).save('./elise_out.mid')

