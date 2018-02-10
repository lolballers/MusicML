import mido
import glob
from tabulate import tabulate

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

dir = './samples'
for file in glob.glob(dir + '/*.mid'):
    print('parsing {}'.format(file))
    mid = mido.MidiFile(file)
    print(mid.ticks_per_beat)
    if mid.type == 0:
        # parse type 0 midi file - 1 track with all info
        tempo = 500000
        us = 0
        events = []
        for event in mid.tracks[0]:
            dt = tempo * event.time / mid.ticks_per_beat
            us += dt
            if event.type == 'set_tempo':
                tempo = event.tempo
            if not event.is_meta:
                events.append((dt / (10**6), us / (10 ** 6), event,))
        print(tabulate(events))
    else:
        raise NotImplementedError('parser only handles type 0 files')

