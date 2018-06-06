import mido
import numpy as np
import glob
import argparse
import os
import shutil
import tqdm
import sklearn.preprocessing
import sys
import typing

"""
Midi IO pipeline
Jens Turner 2018
This version has features for running models by Jens and converting files. The other one is slightly modified for Grant.
Required directory structure (code will not create outer directories):
    - midi_io.py
    - input/
        - event_sequence_v0/
            ... generated event sequences i.e. text-serialized ndarrays
        - midi/
            ... input midi format 0 files
    - output/
        - event_sequence_v0/
            ... writing directory for outputted event sequences
        - midi/
            ... output midi files
"""

parser = argparse.ArgumentParser(allow_abbrev=False)  # for safety
parser.add_argument("--overwrite", help="refresh any files that have already been converted",
                    action="store_true")
parser.add_argument("--delete", help="empty the target directory before parsing any files",
                    action="store_true")
parser.add_argument("direction", help="which direction of data to parse files from",
                    action="store", choices=['input', 'output', 'both'])

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
    assert mid.type == 0  # type 0 files have the least complex timing system, see MIDI standard

    tempo = 500000  # default before any set_tempo according to MIDI standard
    all_events = max(mid.tracks, key=len) # the longest track has the actual music, any others contain metadata
    # filter the events we care about
    all_events = [event for event in all_events if
                  event.type == 'set_tempo' or event.type == 'note_on' or event.type == 'control_change']
    # preallocate output sequence array. set_tempo events do not appear
    sequence_length = sum([1 for event in all_events if event.type == 'note_on' or event.type == 'control_change'])
    event_sequence = np.zeros((sequence_length, 4), dtype=_dtype)

    dt = 0
    i = 0
    for event in all_events:
        dt += tempo * event.time / mid.ticks_per_beat
        if event.type == 'set_tempo':
            tempo = event.tempo
        elif event.type == 'note_on' or event.type == 'control_change':
            # divide dt by 10000 to make range more acceptable for lstm
            if event.type == 'note_on':
                event_sequence[i] = np.array((1, event.note, event.velocity, dt/10000), dtype=_dtype)
            elif event.type == 'control_change':
                event_sequence[i] = np.array((2, event.control, event.value, dt/10000), dtype=_dtype)
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
    # random setup garbage according to midi standard
    mid = mido.MidiFile(ticks_per_beat=_writing_ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=_writing_tempo, time=0))
    track.append(mido.Message('program_change', channel=0, program=0, time=0))
    # multiply time by 10000 to undo scaling from before
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
                                      time=int(event[3] * 10000 * _writing_ticks_per_beat / _writing_tempo)
                                      ))
    return mid

def load_padded_input_event_sequences(basename='*') -> np.ndarray:
    """
    load all input sequences from scratch MIDI and pad to same length with zeros
    :param basename: the file selector
    :return: a 3D numpy array of shape (sequences, max_timesteps, features)
    """
    jagged_sequences = []
    for in_path in glob.glob(os.path.join('.', 'input', 'midi', '{}.mid'.format(basename, _format_version))):
        event_sequence = midi_to_event_sequence(mido.MidiFile(in_path))
        jagged_sequences.append(event_sequence)
    max_len = max(seq.shape[0] for seq in jagged_sequences)
    smooth_sequences = np.zeros((len(jagged_sequences), max_len, 4), dtype=np.float32)
    for i, seq in tqdm.tqdm(enumerate(jagged_sequences), desc="padding into 3D array", file=sys.stdout):
        smooth_sequences[i, :len(seq)] += seq
    return smooth_sequences


def _test_event_sequence_midi():
    # check that transform/inverse-transform/object-model produces the same music
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
    # run the script from command line to convert to/from event sequences
    # if you don't pass any options the script will print the help menu
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
