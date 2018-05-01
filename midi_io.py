import mido
import numpy as np
import itertools
import glob
import argparse
import os
import shutil
import tqdm
import sklearn.preprocessing
import sys
import typing
import operator

parser = argparse.ArgumentParser(allow_abbrev=False)  # for safety
parser.add_argument("--overwrite", help="refresh any files that have already been converted",
                    action="store_true")
parser.add_argument("--delete", help="empty the target directory before parsing any files",
                    action="store_true")
parser.add_argument("direction", help="which direction of data to parse files from",
                    action="store", choices=['in', 'out', 'both'])

# Important Controls:
# 7 = Volume
# 10 = Pan
# 64 = Sustain
# 91 = Reverb

_dtype = np.float32
_writing_ticks_per_beat = 480  # for writing files
_writing_tempo = 500000  # for writing files
_format_version = 0


def midi_to_event_sequence(mid: mido.MidiFile) -> np.ndarray:
    """
    Convert a MIDI file to the event sequence format.
    All times are converted to absolute time in microseconds.
    Events of type note_on and control_change are included in the sequence.
    All other events are ignored.
    :param mid: a mido.MidiFile reference to the file
    :return: an np.ndarray of shape (sequence_length, 4), dtype configured in the header
    """
    assert mid.type <= 1  # no one likes type 2 files, they're dumb

    tempo = 500000  # default before any set_tempo according to MIDI standard
    all_events = max(mid.tracks, key=len)
    all_events = [event for event in all_events if
                  event.type == 'set_tempo' or event.type == 'note_on' or event.type == 'control_change']
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
                event_sequence[i] = np.array((1, event.note, event.velocity, dt/10000), dtype=_dtype)
            elif event.type == 'control_change':
                event_sequence[i] = np.array((2, event.control, event.value, dt), dtype=_dtype)

            i += 1
            dt = 0

    return event_sequence


def event_sequence_to_midi(event_sequence: np.ndarray) -> mido.MidiFile:
    """
    Convert an event sequence to a MIDI file.
    Absolute time is translated to MIDI time with parameters configurable in the header.
    As event sequences only contain note_on and control_change messages, only these are included.
    The file begins with a set_tempo followed by a program_change.
    :param event_sequence: the event sequence
    :return: a mido.MidiFile containing the song
    """

    mid = mido.MidiFile(ticks_per_beat=_writing_ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=_writing_tempo, time=0))
    track.append(mido.Message('program_change', channel=0, program=0, time=0))
    for i, event in enumerate(event_sequence):
        if event[0] == 1:  # note_on
            track.append(mido.Message('note_on',
                                      note=int(event[1]),
                                      velocity=int(event[2]),
                                      time=int(event[3] * 10000 * _writing_ticks_per_beat / _writing_tempo)
                                      ))
        if event[0] == 2:  # control_change
            track.append(mido.Message('control_change',
                                      control=int(event[1]),
                                      value=int(event[2]),
                                      time=int(event[3] * _writing_ticks_per_beat / _writing_tempo)
                                      ))
    mid.save("./generated.mid")
    return mid

# def load_everything():
#     songs = glob.glob(os.path.join('.', 'input', 'midi2', '*.mid'))
#     X = []
#     Y = []
#     for in_path in songs:
#         event_sequence = midi_to_event_sequence(mido.MidiFile(in_path))
#         X.append(event_sequence[0:30])
#         Y.append(np.roll(event_sequence[0:30], 1))
#
#     return np.asarray(X), np.asarray(Y)

def load_everything():
    songs = glob.glob(os.path.join('.', 'input', 'midi2', '*.mid'))
    control = []
    value = []
    velocity = []
    time = []

    for in_path in songs:
        event_sequence = midi_to_event_sequence(mido.MidiFile(in_path))
        for note in event_sequence:
            control.append(note[0])
            value.append(note[1])
            velocity.append(note[2])
            time.append(note[3])
            if(note[3]>1000):
                print(in_path)
                print(note[3])

    return np.asarray(control), np.asarray(value), np.asarray(velocity), np.asarray(time)


def load_padded_input_event_sequences(basename='*') -> typing.Tuple[np.ndarray, sklearn.preprocessing.MinMaxScaler]:
    scaler = sklearn.preprocessing.MinMaxScaler()
    jagged_sequences = []
    for in_path in glob.glob(os.path.join('.', 'input', 'midi', '{}.mid'.format(basename, _format_version))):
        event_sequence = midi_to_event_sequence(mido.MidiFile(in_path))
        scaler.partial_fit(event_sequence)
        jagged_sequences.append(event_sequence)
    # for in_path in tqdm.tqdm(glob.glob(os.path.join('.', 'input', 'event_sequence_v{}'.format(_format_version),
    #                                                 '{}.seq{}'.format(basename, _format_version))),
    #                          desc="loading event sequences", file=sys.stdout):
    #     event_sequence = np.loadtxt(in_path, dtype=_dtype)
    #     scaler.partial_fit(event_sequence)
    #     jagged_sequences.append(event_sequence)
    max_len = max(seq.shape[0] for seq in jagged_sequences)
    smooth_sequences = np.zeros((len(jagged_sequences), max_len, 4), dtype=np.float32)
    for i, seq in tqdm.tqdm(enumerate(jagged_sequences), desc="padding/scaling into 3D 0-1 valued array", file=sys.stdout):
        smooth_sequences[i, :len(seq)] += scaler.transform(seq)
    return smooth_sequences, scaler


def _test_event_sequence_midi():
    mid = mido.MidiFile('./input/midi/alb_esp1.mid')
    event_sequence_to_midi(midi_to_event_sequence(mid)).save('./alb_esp1_out.mid')
    mid = mido.MidiFile('./alb_esp1_out.mid')
    mid.save('./alb_esp1_out2.mid')


def _empty_directory(folder):
    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


if __name__ == '__main__':
    flags = parser.parse_args()

    if flags.direction == 'input' or flags.direction == 'both':
        if flags.delete:
            _empty_directory(os.path.join('.', 'input', 'event_sequence_v{0}'.format(_format_version)))

        for in_path in glob.glob(os.path.join('.', 'input', 'midi', '*.mid')):
            out_path = os.path.join('.', 'input', 'event_sequence_v{0}'.format(_format_version),
                                    '{}.seq{}'.format(os.path.splitext(os.path.basename(in_path))[0], _format_version))

            if flags.overwrite or not os.path.exists(out_path):
                mid = mido.MidiFile(in_path)
                event_sequence = midi_to_event_sequence(mid)
                np.savetxt(out_path, event_sequence)

    if flags.direction == 'output' or flags.direction == 'both':
        if flags.delete:
            _empty_directory(os.path.join('.', 'output', 'midi'))

        for in_path in glob.glob(os.path.join('.', 'output', 'event_sequence_v{}'.format(_format_version),
                                              '*.seq{}'.format(_format_version))):
            out_path = os.path.join('.', 'output', 'midi',
                                    '{}.mid'.format(os.path.splitext(os.path.basename(in_path))[0]))

            if flags.overwrite or not os.path.exists(out_path):
                event_sequence = np.loadtxt(in_path, dtype=_dtype)
                mid = event_sequence_to_midi(event_sequence)
                mid.save(out_path)
