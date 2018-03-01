import mido
import numpy as np

from unused import midi_manipulation

# NOTES ABOUT FORMATTING
# Right now, we only use MIDI type 0 files, which contain all messages in 1 track.
# That way, the tracks don't have to be merged when computing absolute time.
# At some point in the future, this should be extended to types 1/2.
#
# For re-encoding, the header block from piano-midi.de looks like this:
# (0.0, <message program_change channel=0 program=0 time=0>)
# (0.0, <message control_change channel=0 control=7 value=100 time=0>)
# (0.0, <message control_change channel=0 control=10 value=64 time=0>)
# (0.0, <message control_change channel=0 control=91 value=127 time=0>)
# (0.0, <message program_change channel=0 program=0 time=0>)
# (0.0, <message control_change channel=0 control=7 value=100 time=0>)
# (0.0, <message control_change channel=0 control=10 value=64 time=0>)
#
# Important Controls:
# 7 = Volume
# 10 = Pan
# 91 = Reverb
# 64 = Sustain
#
# Piano-midi.de also represents <note_off> messages as <note_on velocity=0>.
# Apparently some keyboards do this by default.
# In my opinion, this is dumb, and the re-encoder should use actual <note_off>s.


_num_notes = 128 # in midi standard
_dt = 0.001 # seconds/frame
_dtype = np.bool


def midi_to_note_state_matrix(mid: mido.MidiFile) -> np.ndarray:
    if mid.type > 1:
        raise NotImplementedError('parser only handles type 0-1 files')
    print(mid.ticks_per_beat)
    # parse type 0 midi file - 1 track with all info
    # or type 1 - several tracks played in sequence
    # Build an absolute-time keyed list of every event.
    tempo = 500000 # default tempo according to MIDI standard
    us = 0
    events = []
    for event in np.concatenate(tuple(mid.tracks)):
        dt = tempo * event.time / mid.ticks_per_beat
        us += dt
        if event.type == 'set_tempo':
            tempo = event.tempo
            #print(event.tempo)
        if not event.is_meta:
            events.append((us / (10 ** 6), event))
    # print(tabulate(events))

    print(len(events))

    i = 0
    j = 0
    sustain_on = False
    # range of values from 0 to final time rounded to nearest dt
    internal_state = np.full(_num_notes, 0, dtype=_dtype)

    matrix_length = len(np.arange(0, events[-1][0], _dt))
    note_state_matrix = np.full((matrix_length, _num_notes), 0, dtype=_dtype)

    for t in range(matrix_length):
        # Start by checking the events list to see whether there are any events in this frame.
        while (i + 1) < len(events) and events[i][0] < (t + _dt):
            event = events[i][1]

            if event.type == 'control_change' and event.control == 64:
                sustain_on = (event.value > 0)
            elif event.type == 'note_on':
                if event.velocity > 0:
                    j += 1
                internal_state[event.note] = (event.velocity > 0)

            i += 1

        # if sustain_on and t > 0:
            # no previous state at time step 0; previous state is ignored for no sustain
            # note_state_matrix[t] = np.bitwise_or(np.copy(note_state_matrix[t - 1]), internal_state)
        # else:
        note_state_matrix[t] = internal_state
    print(j)
    return note_state_matrix


def note_state_matrix_to_midi(note_state_matrix: np.ndarray) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', channel=0, program=0, time=0))
    track.append(mido.Message('control_change', channel=0, control=7, value=100, time=0))
    track.append(mido.Message('control_change', channel=0, control=10, value=64, time=0))
    track.append(mido.Message('control_change', channel=0, control=91, value=127, time=0))
    track.append(mido.MetaMessage('set_tempo', tempo=800000, time=0))

    j = 0
    ct = 0
    internal_state = np.full(_num_notes, 0, dtype=_dtype)
    for note_state in note_state_matrix:
        for i, note in enumerate(note_state):
            if note and not internal_state[i]:
                track.append(mido.Message('note_on', channel=0, note=i, velocity=64, time=ct))
                j += 1
                ct = 0
            elif not note and internal_state[i]:
                track.append(mido.Message('note_off', channel=0, note=i, time=ct))
                j += 1
                ct = 0
        ct += 480
        internal_state = note_state
    print(j)

    return mid

mid = mido.MidiFile('./samples/midi/elise_format0.mid')
note_state_matrix_to_midi(midi_to_note_state_matrix(mid)).save('./elise_out.mid')

midi_manipulation.noteStateMatrixToMidi(midi_manipulation.midiToNoteStateMatrix('./samples/midi/elise_format0.mid'), name='./elise_out_eli.mid')
