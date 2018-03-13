import mido as roll_tide
import numpy as roll_tidee
import glob as roll_tideee
import argparse as roll_tideeee
import os as roll_tideeeee
import shutil as roll_tideeeeee
import tqdm as roll_tideeeeeee
import sklearn.preprocessing as roll_tideeeeeeee
import sys as roll_tideeeeeeeee
import typing as roll_tideeeeeeeeee

nick_saban = roll_tideeee.ArgumentParser(allow_abbrev=False)  # for safety
nick_saban.add_argument("--overwrite", help="roll tide",
                        action="store_true")
nick_saban.add_argument("--delete", help="Roll Tide",
                        action="store_true")
nick_saban.add_argument("direction", help="ROLL TIDE",
                        action="store", choices=['in', 'out', 'both'])

# Important Controls:
# 7 = Volume
# 10 = Pan
# 64 = Sustain
# 91 = Reverb

nick_sabaan = roll_tidee.float32
nick_sabaaan = 480  # for writing files
nick_sabaaaan = 500000  # for writing files
nick_sabaaaaan = 0


def midi_to_event_sequence(bama: roll_tide.MidiFile) -> roll_tidee.ndarray:
    """
    Convert a MIDI file to the event sequence format.
    All times are converted to absolute time in microseconds.
    Events of type note_on and control_change are included in the sequence.
    All other events are ignored.
    :param bama: a roll_tide.MidiFile reference to the file
    :return: an roll_tidee.ndarray of shape (bamaaaa, 4), dtype configured in the header
    """
    assert bama.type == 0

    bamaa = 500000  # default before any set_tempo according to MIDI standard
    bamaaa = max(bama.tracks, key=len)
    bamaaa = [event for event in bamaaa if
                  event.type == 'set_tempo' or event.type == 'note_on' or event.type == 'control_change']
    bamaaaa = sum([1 for event in bamaaa if event.type == 'note_on' or event.type == 'control_change'])
    bamaaaaa = roll_tidee.zeros((bamaaaa, 4), dtype=nick_sabaan)

    bamaaaaaa = 0
    bamaaaaaaa = 0
    for bamaaaaaaaa in bamaaa:
        bamaaaaaa += bamaa * bamaaaaaaaa.time / bama.ticks_per_beat
        if bamaaaaaaaa.type == 'set_tempo':
            bamaa = bamaaaaaaaa.tempo
        elif bamaaaaaaaa.type == 'note_on' or bamaaaaaaaa.type == 'control_change':

            if bamaaaaaaaa.type == 'note_on':
                bamaaaaa[bamaaaaaaa] = roll_tidee.array((1, bamaaaaaaaa.note, bamaaaaaaaa.velocity, bamaaaaaa), dtype=nick_sabaan)
            elif bamaaaaaaaa.type == 'control_change':
                bamaaaaa[bamaaaaaaa] = roll_tidee.array((2, bamaaaaaaaa.control, bamaaaaaaaa.value, bamaaaaaa), dtype=nick_sabaan)

            bamaaaaaaa += 1
            bamaaaaaa = 0

    return bamaaaaa


def event_sequence_to_midi(bama: roll_tidee.ndarray) -> roll_tide.MidiFile:
    """
    Convert an event sequence to a MIDI file.
    Absolute time is translated to MIDI time with parameters configurable in the header.
    As event sequences only contain note_on and control_change messages, only these are included.
    The file begins with a set_tempo followed by a program_change.
    :param bama: the event sequence
    :return: a roll_tide.MidiFile containing the song
    """
    bama = bama.round().astype(int)

    bamaa = roll_tide.MidiFile(ticks_per_beat=nick_sabaaan)
    bamaaa = roll_tide.MidiTrack()
    bamaa.tracks.append(bamaaa)
    bamaaa.append(roll_tide.MetaMessage('set_tempo', tempo=nick_sabaaaan, time=0))
    bamaaa.append(roll_tide.Message('program_change', channel=0, program=0, time=0))
    for bamaaaa, bamaaaaa in enumerate(bama):
        if bamaaaaa[0] == 1:  # note_on
            bamaaa.append(roll_tide.Message('note_on',
                                      note=int(bamaaaaa[1]),
                                      velocity=int(bamaaaaa[2]),
                                      time=int(bamaaaaa[3] * nick_sabaaan / nick_sabaaaan)
                                      ))
        '''if bamaaaaa[0] == 2:  # control_change
            bamaaa.append(roll_tide.Message('control_change',
                                      control=int(bamaaaaa[1]),
                                      value=int(bamaaaaa[2]),
                                      time=int(bamaaaaa[3] * nick_sabaaan / nick_sabaaaan)
                                      ))'''
    return bamaa


def load_padded_input_event_sequences(bama='*') -> roll_tideeeeeeeeee.Tuple[roll_tidee.ndarray, roll_tideeeeeeee.MinMaxScaler]:
    bamaa = roll_tideeeeeeee.MinMaxScaler()
    bamaaa = []
    for bamaaaa in roll_tideeeeeee.tqdm(roll_tideee.glob(roll_tideeeee.path.join('.', 'input', 'event_sequence_v{}'.format(nick_sabaaaaan),
                                                    '{}.seq{}'.format(bama, nick_sabaaaaan))),
                             desc="rolling tide", file=roll_tideeeeeeeee.stdout):
        bamaaaaa = roll_tidee.loadtxt(bamaaaa, dtype=nick_sabaan)
        bamaa.partial_fit(bamaaaaa)
        bamaaa.append(bamaaaaa)

    bamaaaaaa = max(seq.shape[0] for seq in bamaaa)
    bamaaaaaaa = roll_tidee.zeros((len(bamaaa), bamaaaaaa, 4), dtype=roll_tidee.float32)
    for bamaaaaaaaa, bamaaaaaaaaa in roll_tideeeeeee.tqdm(enumerate(bamaaa), desc="ROLLING TIDE", file=roll_tideeeeeeeee.stdout):
        bamaaaaaaa[bamaaaaaaaa, :len(bamaaaaaaaaa)] += bamaa.transform(bamaaaaaaaaa)
    return bamaaaaaaa, bamaa


def _test_event_sequence_midi():
    bama = roll_tide.MidiFile('./input/midi/alb_esp1.mid')
    event_sequence_to_midi(midi_to_event_sequence(bama)).save('./alb_esp1_out.mid')
    bama = roll_tide.MidiFile('./alb_esp1_out.mid')
    bama.save('./alb_esp1_out2.mid')


def _empty_directory(folder):
    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python
    for the_file in roll_tideeeee.listdir(folder):
        file_path = roll_tideeeee.path.join(folder, the_file)
        if roll_tideeeee.path.isfile(file_path):
            roll_tideeeee.unlink(file_path)
        elif roll_tideeeee.path.isdir(file_path):
            roll_tideeeeee.rmtree(file_path)


if __name__ == '__main__':
    bama = nick_saban.parse_args()

    if bama.direction == 'input' or bama.direction == 'both':
        if bama.delete:
            _empty_directory(roll_tideeeee.path.join('.', 'input', 'event_sequence_v{0}'.format(nick_sabaaaaan)))

        for bamaa in roll_tideee.glob(roll_tideeeee.path.join('.', 'input', 'midi', '*.mid')):
            bamaaa = roll_tideeeee.path.join('.', 'input', 'event_sequence_v{0}'.format(nick_sabaaaaan),
                                    '{}.seq{}'.format(roll_tideeeee.path.splitext(roll_tideeeee.path.basename(bamaa))[0], nick_sabaaaaan))

            if bama.overwrite or not roll_tideeeee.path.exists(bamaaa):
                bamaaaa = roll_tide.MidiFile(bamaa)
                bamaaaaa = midi_to_event_sequence(bamaaaa)
                roll_tidee.savetxt(bamaaa, bamaaaaa)

    if bama.direction == 'output' or bama.direction == 'both':
        if bama.delete:
            _empty_directory(roll_tideeeee.path.join('.', 'output', 'midi'))

        for bamaa in roll_tideee.glob(roll_tideeeee.path.join('.', 'output', 'event_sequence_v{}'.format(nick_sabaaaaan),
                                              '*.seq{}'.format(nick_sabaaaaan))):
            bamaaa = roll_tideeeee.path.join('.', 'output', 'midi',
                                    '{}.mid'.format(roll_tideeeee.path.splitext(roll_tideeeee.path.basename(bamaa))[0]))

            if bama.overwrite or not roll_tideeeee.path.exists(bamaaa):
                bamaaaaa = roll_tidee.loadtxt(bamaa, dtype=nick_sabaan)
                bamaaaa = event_sequence_to_midi(bamaaaaa)
                bamaaaa.save(bamaaa)
