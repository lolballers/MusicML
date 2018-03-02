import mido
import numpy as np
import itertools
import glob
import argparse
import os
import shutil

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
    all_events = itertools.chain.from_iterable(mid.tracks)
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
                event_sequence[i] = np.array((1, event.note, event.velocity, dt), dtype=_dtype)
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
                                      time=int(event[3] * _writing_ticks_per_beat / _writing_tempo)
                                      ))
        if event[0] == 2:  # control_change
            track.append(mido.Message('control_change',
                                      control=int(event[1]),
                                      value=int(event[2]),
                                      time=int(event[3] * _writing_ticks_per_beat / _writing_tempo)
                                      ))
    return mid


def _test_event_sequence_midi():
    mid = mido.MidiFile('./input/midi/elise_format0.mid')
    event_sequence_to_midi(midi_to_event_sequence(mid)).save('./elise_out.mid')


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
