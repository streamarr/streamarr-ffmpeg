#!/usr/bin/env python3
"""Checks the DVD PCM that FFmpeg writes into MPEG-PS, and reads back after a seek.

    python3 streamarr/tests/dvd_lpcm_mpegps.py --ffmpeg PATH

FFmpeg's DVD PCM decoder reads 20 and 24 bit samples in blocks: the upper 16 bits of four samples,
then their lower bits. A block holds those four samples for one, two or four channels, two groups
of four for eight channels, and four samples of every channel otherwise. A decode that starts after
a seek takes the first byte of its first packet as the start of a block, so every packet that a
seek can land on must start on one. 16-bit samples need only whole sample frames.

Each source is counted PCM beside MPEG-2 video, written by FFmpeg as a VOB. Sample n of channel c
holds n + 1, modulo the counter's range, in its lower bits and c in its top three bits, so every
decoded row (one sample of every channel) names the source sample it came from. The check reads
the LPCM packets of the written file and decodes it from the start and after each seek:

    every LPCM payload starts on a block of the decoder
    the decode from the start holds every source sample, in order
    each decode after a seek holds only source samples, each one following the one before

DVD-Video gives each LPCM packet the number of audio frames that start in its payload and a pointer
to the first of them, where 4 is the payload's first byte. The packet's timestamp names that frame.
The check also requires:

    every packet with a timestamp starts a frame at its pointer, and the timestamp is that byte's
    time, to within one tick of the 90 kHz clock
    a packet in which no frame starts carries no timestamp
    every decoded frame, from the start and after each seek, is stamped with the time of the source
    sample that it starts with, to within one sample

The third byte of the LPCM header sets dynamic range control, 0x80 for none. FFmpeg's muxer writes
0x80 into every packet, and a disc can change it from packet to packet. The decode checks also run
on copies of two sources in which every second LPCM packet after the first carries 0x55.

The first byte of the first LPCM packet can be before zero while the video, read first, starts after
it. The check also decodes copies of two sources whose LPCM is moved there, and requires every frame
to be stamped with the time of the source sample it starts with, before zero too.

It also writes 24-bit 5.1, whose block is 72 bytes, alone into packets too small for a block. A
VOB packet of 96 bytes has room for 60 after the pack header that every MPEG-2 packet carries, a
PES header with a PTS and the LPCM header. An MPEG-1 system stream packet of 88 bytes has room for
70 without a pack header. No block fits, so FFmpeg must refuse those packet sizes. A 128-byte
packet has room for one block even with a pack header, a PTS and a DTS, so FFmpeg must write it.
Every size between is refused or written in whole blocks, and FFmpeg must stop: a muxer that
cannot write a block writes empty packets forever.

Last, it writes 24-bit DVD PCM alone into VOBs: 5.1 that FFmpeg encodes and 7 channels that
FFmpeg copies, in packets of the default 2048 bytes and of 125 to 3000 bytes. Whole blocks leave
up to a block of a packet unused, and the headers of a small packet take more than the 5 % that
FFmpeg adds to the mux rate for them, so FFmpeg must raise the mux rate, which it computes from the
stream's bit rate, to carry them. The pack that completes a block must start to arrive, by its
SCR, before the block's first sample is due. A block that arrives later leaves a player that keeps
to that clock without the samples it must play.

Exits 0 when every check passes, 1 when one fails, 2 when FFmpeg fails.
"""
import argparse
import array
import collections
import os
import resource
import subprocess
import sys
import tempfile

SECONDS = 6
SEEKS = (1.3, 2.6, 3.9, 5.2)
# bits, rate, channels and encoder frame size of every shape FFmpeg's encoder writes under
# DVD-Video's 9.8 Mb/s, with the encoder's own frames (0) beside video. The last shape's frames of 120
# samples, 2,160 bytes, are longer than a packet's payload, so some packets hold no frame's start. It
# has no video: with video, the order in which FFmpeg hands the muxer its packets varies from run to
# run, and with it which packets hold no frame's start.
SHAPES = [(16, 48000, 1, 0), (16, 48000, 2, 0), (16, 48000, 6, 0), (16, 96000, 6, 0),
          (24, 48000, 1, 0), (24, 48000, 2, 0), (24, 48000, 6, 0), (24, 96000, 1, 0),
          (24, 96000, 2, 0), (24, 48000, 6, 120)]
# The shapes whose packets also get a dynamic range byte that changes from packet to packet.
VARYING_DYNAMIC_RANGE_SHAPES = [(16, 48000, 2, 0), (24, 48000, 2, 0)]
# The shapes also read with their LPCM moved to start before zero, beside video that starts after it.
START_BEFORE_ZERO_SHAPES = [(16, 48000, 2, 0), (24, 48000, 2, 0)]
LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1"}
SMALL_PACKETS = {"vob": range(96, 129), "mpeg": range(88, 129)}
FORMAT_NAMES = {"vob": "a VOB", "mpeg": "an MPEG-1 system stream"}
SMALL_PACKET_SAMPLES = 24000
# Half a second of 24-bit 5.1 fills about 0.7 MB of these packets. A muxer that never drains its
# queue writes past this limit within a second, and the limit stops it.
OUTPUT_LIMIT = 16 << 20
# The default, then from the smallest packet with room for a block of 7 channels after a pack
# header, a PTS and a DTS, in a step prime to the 72-byte block of 24-bit 5.1 and the 84-byte block
# of 7 channels, so the bytes that whole blocks leave unused take every value below a block. Above
# about 3500 bytes the 4 KB audio buffer that the muxer models holds each packet back until its
# first block is all but due.
ARRIVAL_PACKET_SIZES = [2048] + list(range(125, 3001, 11))


def block_size(bits, channels):
    """The bytes of one block of the decoder (libavcodec/pcm-dvd.c)."""
    if bits == 16:
        return 2 * channels
    if channels in (1, 2, 4):
        return 4 * bits // 8
    if channels == 8:
        return 8 * bits // 8
    return 4 * channels * bits // 8


def counted(bits, channels, count):
    """Signed 32-bit little-endian PCM whose upper bits hold each sample's count and channel."""
    period = 1 << (bits - 3)
    samples = array.array("i", bytes(4 * count * channels))
    index = 0
    for n in range(count):
        for channel in range(channels):
            word = ((((n + 1) % period) | channel << (bits - 3)) << (32 - bits)) & 0xFFFFFFFF
            samples[index] = word - (1 << 32) if word >> 31 else word
            index += 1
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        sys.stderr.write("FFmpeg failed: %s\n" % " ".join(command))
        sys.exit(2)
    return result.stdout


def limit_output():
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT, OUTPUT_LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def exit_status_within_limits(command):
    """FFmpeg's exit status, or None when it wrote past OUTPUT_LIMIT or ran for a minute."""
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=limit_output, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    return result.returncode if result.returncode >= 0 else None


def write_source(ffmpeg, directory, shape):
    bits, rate, channels, frame_size = shape
    raw = os.path.join(directory, "b%d-r%d-c%d-f%d.s32le" % shape)
    vob = os.path.join(directory, "b%d-r%d-c%d-f%d.vob" % shape)
    with open(raw, "wb") as out:
        out.write(counted(bits, channels, rate * SECONDS))
    video = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=%d" % SECONDS]
    video_codec = ["-c:v", "mpeg2video", "-g", "12", "-bf", "0"]
    audio_codec = ["-c:a", "pcm_dvd", "-sample_fmt", "s16" if bits == 16 else "s32"]
    if frame_size:
        video = video_codec = []
        audio_codec += ["-frame_size", str(frame_size)]
    run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"] + video
        + ["-f", "s32le", "-ar", str(rate), "-ch_layout", LAYOUTS[channels], "-i", raw]
        + video_codec + audio_codec + ["-f", "vob", vob])
    return vob


def timestamp(data, position):
    """The 33-bit PTS or MPEG-1 SCR in the five bytes at position, in 90 kHz ticks."""
    b = data[position:position + 5]
    return ((b[0] >> 1) & 7) << 30 | b[1] << 22 | (b[2] >> 1) << 15 | b[3] << 7 | b[4] >> 1


def set_timestamp(data, offset, value):
    """Writes a 33-bit PES timestamp into the five bytes at offset, keeping their first four bits."""
    field = pes_timestamp(value)
    data[offset:offset + 5] = bytes([data[offset] & 0xF0 | field[0] & 0x0F]) + field[1:]


def pack_scr(data, position):
    """The size of the MPEG-2 or MPEG-1 pack header at position, and its SCR in 90 kHz ticks."""
    if data[position + 4] & 0xC0 == 0x40:
        b = data[position + 4:position + 10]
        base = (((b[0] >> 3) & 7) << 30 | (b[0] & 3) << 28 | b[1] << 20 | (b[2] >> 3) << 15
                | (b[2] & 3) << 13 | b[3] << 5 | b[4] >> 3)
        extension = (b[4] & 3) << 7 | b[5] >> 1
        return 14 + (data[position + 13] & 7), base + extension / 300
    return 12, timestamp(data, position + 4)


def pes_body(data, position):
    """Where the data of the PES packet at position starts, after an MPEG-2 or MPEG-1 header, and
    the packet's PTS, or None."""
    if data[position + 6] & 0xC0 == 0x80:
        pts = timestamp(data, position + 9) if data[position + 7] & 0x80 else None
        return position + 9 + data[position + 8], pts
    body = position + 6
    while data[body] == 0xFF:
        body += 1
    if data[body] & 0xC0 == 0x40:
        body += 2
    timestamps = data[body] & 0xF0
    if timestamps == 0x20:
        return body + 5, timestamp(data, body)
    if timestamps == 0x30:
        return body + 10, timestamp(data, body)
    return body + 1, None


# scr: the SCR of the packet's pack, when the pack starts to reach the decoder (ISO/IEC 13818-1,
# 2.5.2). No byte of the packet arrives earlier.
LpcmPacket = collections.namedtuple("LpcmPacket", "frames pointer payload pts scr")


def pes_packets(data):
    """The (PES packet start, stream id, PES packet end, SCR of its pack) of every PES packet of the
    bytes of an MPEG-2 program stream or an MPEG-1 system stream."""
    position = 0
    scr = None
    while position + 4 <= len(data):
        if data[position:position + 3] != b"\x00\x00\x01":
            raise ValueError("no start code at byte %d" % position)
        stream_id = data[position + 3]
        if stream_id == 0xB9:
            break
        if stream_id == 0xBA:
            pack_header, scr = pack_scr(data, position)
            position += pack_header
            continue
        end = position + 6 + int.from_bytes(data[position + 4:position + 6], "big")
        yield position, stream_id, end, scr
        position = end


def lpcm_packet_bounds(data):
    """The (PES packet start, LPCM sub-stream start, PES packet end, PTS or None, SCR of its pack)
    of every LPCM packet of the bytes of an MPEG-2 program stream or an MPEG-1 system stream."""
    for position, stream_id, end, scr in pes_packets(data):
        if stream_id == 0xBD:
            body, pts = pes_body(data, position)
            if 0xA0 <= data[body] <= 0xAF:
                yield position, body, end, pts, scr


def lpcm_packets(vob):
    """Every LPCM packet, in order, of an MPEG-2 program stream or an MPEG-1 system stream."""
    with open(vob, "rb") as source:
        data = source.read()
    packets = []
    for _, body, end, pts, scr in lpcm_packet_bounds(data):
        packets.append(LpcmPacket(
            frames=data[body + 1], pointer=int.from_bytes(data[body + 2:body + 4], "big"),
            payload=data[body + 7:end], pts=pts, scr=scr))
    return packets


def vary_dynamic_range(vob):
    """A copy of the file in which every second LPCM packet after the first carries dynamic range
    byte 0x55. The first keeps 0x80: FFmpeg's demuxer takes the stream for DVD PCM by it."""
    with open(vob, "rb") as source:
        data = bytearray(source.read())
    for index, (_, body, _, _, _) in enumerate(lpcm_packet_bounds(data)):
        if index % 2:
            data[body + 6] = 0x55
    varied = vob[:-len(".vob")] + "-varying-dynamic-range.vob"
    with open(varied, "wb") as out:
        out.write(data)
    return varied


def start_before_zero(vob):
    """A copy of the file whose LPCM starts before zero, and the bytes in front of the first frame
    of its first LPCM packet.

    The copy keeps the LPCM from the first packet after the first stamped video packet in which a
    frame starts past the payload's first byte, turns the LPCM packets before it into padding, and
    moves every LPCM timestamp back so that this frame is stamped 0. The video keeps its
    timestamps: FFmpeg's demuxer reads it first and takes its timestamp wrap reference from it."""
    with open(vob, "rb") as source:
        data = bytearray(source.read())
    first_video = next(position for position, stream_id, _, _ in pes_packets(data)
                       if 0xE0 <= stream_id <= 0xEF and data[position + 7] & 0x80)
    shift = lead = None
    for position, body, end, _, _ in lpcm_packet_bounds(data):
        # FFmpeg's muxer writes no DTS into an LPCM packet, so moving the PTS moves the packet.
        stamped = data[position + 7] & 0x80
        pointer = int.from_bytes(data[body + 2:body + 4], "big")
        if shift is None and position > first_video and stamped and pointer > 4:
            shift, lead = timestamp(data, position + 9), pointer - 4
        if shift is None:
            data[position + 3] = 0xBE
            data[position + 6:end] = b"\xff" * (end - position - 6)
        elif stamped:
            set_timestamp(data, position + 9, timestamp(data, position + 9) - shift)
    if lead is None:
        raise ValueError("no LPCM packet after the first video packet starts a frame past its"
                         " payload's first byte")
    moved = vob[:-len(".vob")] + "-before-zero.vob"
    with open(moved, "wb") as out:
        out.write(data)
    return moved, lead


def misaligned_payloads(vob, bits, channels):
    block = block_size(bits, channels)
    offset = misaligned = 0
    packets = lpcm_packets(vob)
    for packet in packets:
        misaligned += offset % block != 0
        offset += len(packet.payload)
    return misaligned, len(packets)


def misplaced_timestamps(vob, bits, rate, channels):
    """Packets whose timestamp is not the time of the byte at their pointer, of all stamped ones.

    The first packet's first byte is the stream's first byte, so its time sets the zero."""
    ticks_per_byte = 90000 * 8 / (rate * channels * bits)
    origin = None
    offset = misplaced = stamped = 0
    for packet in lpcm_packets(vob):
        if packet.pts is not None:
            stamped += 1
            if packet.frames == 0 or not 4 <= packet.pointer < len(packet.payload) + 4:
                misplaced += 1
            else:
                first_byte = offset + packet.pointer - 4
                if origin is None:
                    origin = packet.pts - first_byte * ticks_per_byte
                misplaced += abs(origin + first_byte * ticks_per_byte - packet.pts) > 1
        offset += len(packet.payload)
    return misplaced, stamped


def decode(ffmpeg, vob, shape, seek):
    """Each decoded row as its source sample index, or None for a row that names none, and each
    decoded frame as (stamp in samples on the source's clock, first row, rows)."""
    bits, rate, channels, _ = shape
    seek_arguments = ["-ss", str(seek)] if seek is not None else []
    raw = vob + ".s32le"
    frames = run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-copyts"] + seek_arguments
                 + ["-i", vob, "-map", "0:a:0", "-c:a", "pcm_s32le", "-f", "s32le", "-y", raw,
                    "-map", "0:a:0", "-c:a", "pcm_s32le", "-f", "framecrc", "-"])
    with open(raw, "rb") as decoded:
        pcm = decoded.read()
    stamps = []
    row = 0
    for line in frames.decode().splitlines():
        if line.startswith("#tb 0:"):
            numerator, denominator = (int(part) for part in line.split()[-1].split("/"))
        elif line and not line.startswith("#"):
            fields = [field.strip() for field in line.split(",")]
            count = int(fields[4]) // (4 * channels)
            stamps.append((int(fields[2]) * numerator * rate // denominator, row, count))
            row += count
    return decoded_rows(pcm, bits, channels), stamps


def decoded_rows(pcm, bits, channels):
    words = array.array("I", pcm[:len(pcm) - len(pcm) % (4 * channels)])
    if sys.byteorder == "big":
        words.byteswap()
    period = 1 << (bits - 3)
    rows = []
    for start in range(0, len(words), channels):
        values = [word >> (32 - bits) for word in words[start:start + channels]]
        counts = {value & (period - 1) for value in values}
        tags = [value >> (bits - 3) for value in values]
        if len(counts) != 1 or tags != list(range(channels)):
            rows.append(None)
            continue
        rows.append((counts.pop() - 1) % period)
    return rows


def rows_out_of_sequence(rows, period):
    """The rows that hold no source sample or do not follow the row before."""
    wrong = 0
    previous = None
    for row in rows:
        if row is None or (previous is not None and row != (previous + 1) % period):
            wrong += 1
        previous = row
    return wrong


def stamp_errors(stamps, rows, origin, period):
    """Each frame's stamp less the time of the source sample it starts with, in samples, for the
    frames whose first row names a source sample."""
    errors = []
    for stamp, first, _ in stamps:
        if first < len(rows) and rows[first] is not None:
            errors.append((stamp - origin - rows[first] + period // 2) % period - period // 2)
    return errors


def report_stamps(name, errors):
    right = bool(errors) and max(abs(error) for error in errors) <= 1
    print("%s: %d decoded frames stamped %s samples from their first source sample: %s"
          % (name, len(errors), "%+d to %+d" % (min(errors), max(errors)) if errors else "-",
             "ok" if right else "WRONG"))
    return not right


def shape_name(shape):
    bits, rate, channels, frame_size = shape
    name = "%d-bit %d Hz %s" % (bits, rate, LAYOUTS[channels])
    if frame_size:
        name += " in frames of %d samples" % frame_size
    return name


def check_shape(ffmpeg, directory, shape):
    bits, rate, channels, _ = shape
    name = shape_name(shape)
    vob = write_source(ffmpeg, directory, shape)
    failures = 0

    misaligned, total = misaligned_payloads(vob, bits, channels)
    failures += misaligned != 0
    print("%s: %d of %d LPCM payloads start inside a %d-byte block: %s"
          % (name, misaligned, total, block_size(bits, channels), "ok" if not misaligned else "WRONG"))

    misplaced, stamped = misplaced_timestamps(vob, bits, rate, channels)
    failures += misplaced != 0
    print("%s: %d of %d LPCM timestamps do not name the frame at the packet's pointer: %s"
          % (name, misplaced, stamped, "ok" if not misplaced else "WRONG"))
    return failures + check_decodes(ffmpeg, vob, shape, name)


def check_varying_dynamic_range(ffmpeg, directory, shape):
    vob = vary_dynamic_range(write_source(ffmpeg, directory, shape))
    return check_decodes(ffmpeg, vob, shape, shape_name(shape) + " with a varying dynamic range byte")


def check_start_before_zero(ffmpeg, directory, shape):
    """Decodes LPCM whose first byte is before zero, beside video that starts after it."""
    bits, _, channels, _ = shape
    name = shape_name(shape) + ", starting before zero beside video,"
    vob, lead = start_before_zero(write_source(ffmpeg, directory, shape))
    rows, stamps = decode(ffmpeg, vob, shape, None)
    wrong = rows_out_of_sequence(rows, 1 << (bits - 3))
    right = rows and wrong == 0
    print("%s from the start: %d of %d rows hold no source sample or do not follow the one before: %s"
          % (name, wrong, len(rows), "ok" if right else "WRONG"))
    # The first decoded row is the first byte of the first LPCM packet, lead bytes before zero.
    start = -lead * 8 // (channels * bits)
    errors = [stamp - start - first for stamp, first, _ in stamps]
    return (not right) + report_stamps(name + " from the start", errors)


def check_decodes(ffmpeg, vob, shape, name):
    """Decodes the file from the start and after each seek, and checks every row and frame stamp."""
    bits, rate, _, _ = shape
    period = 1 << (bits - 3)
    failures = 0
    rows, stamps = decode(ffmpeg, vob, shape, None)
    expected = rate * SECONDS
    whole = rows == [n % period for n in range(expected)]
    failures += not whole
    print("%s from the start: %d rows for %d source samples, %d out of sequence: %s"
          % (name, len(rows), expected, rows_out_of_sequence(rows, period), "ok" if whole else "WRONG"))
    origin = stamps[0][0]
    failures += report_stamps(name + " from the start", stamp_errors(stamps, rows, origin, period))

    for seek in SEEKS:
        rows, stamps = decode(ffmpeg, vob, shape, seek)
        wrong = rows_out_of_sequence(rows, period)
        right = rows and wrong == 0
        failures += not right
        print("%s after a seek to %.1f s: %d of %d rows hold no source sample or do not follow"
              " the one before: %s" % (name, seek, wrong, len(rows), "ok" if right else "WRONG"))
        failures += report_stamps("%s after a seek to %.1f s" % (name, seek),
                                  stamp_errors(stamps, rows, origin, period))
    return failures


def written_in_whole_blocks(ffmpeg, vob, shape):
    """Whether every payload starts on a block and the decode from the start holds every source
    sample in order, and a description of what the file holds."""
    bits, _, channels, _ = shape
    period = 1 << (bits - 3)
    misaligned, total = misaligned_payloads(vob, bits, channels)
    rows, _ = decode(ffmpeg, vob, shape, None)
    whole = rows == [n % period for n in range(SMALL_PACKET_SAMPLES)]
    return misaligned == 0 and whole, (
        "written, %d of %d LPCM payloads start inside a block, %d of %d rows from the start out of"
        " sequence" % (misaligned, total, rows_out_of_sequence(rows, period), len(rows)))


def expected_outcome(packet_sizes, packet_size, refused, whole):
    """What FFmpeg must do with one of packet_sizes, and whether it did."""
    if packet_size == packet_sizes[0]:
        return "refused, as no block fits", refused
    if packet_size == packet_sizes[-1]:
        return "written in whole blocks, as a block fits", whole
    return "refused, or written in whole blocks", refused or whole


def check_small_packets(ffmpeg, directory):
    bits, rate, channels = 24, 48000, 6
    raw = os.path.join(directory, "small-packets.s32le")
    with open(raw, "wb") as out:
        out.write(counted(bits, channels, SMALL_PACKET_SAMPLES))
    failures = 0
    for output_format, packet_sizes in SMALL_PACKETS.items():
        for packet_size in packet_sizes:
            path = os.path.join(directory, "small-packets-%d.%s" % (packet_size, output_format))
            status = exit_status_within_limits(
                [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "s32le", "-ar", str(rate), "-ch_layout", LAYOUTS[channels], "-i", raw,
                 "-c:a", "pcm_dvd", "-sample_fmt", "s32", "-packetsize", str(packet_size),
                 "-f", output_format, path])
            refused = status is not None and status != 0
            whole = False
            outcome = "refused"
            if status is None:
                outcome = "FFmpeg did not stop"
            elif not refused:
                whole, outcome = written_in_whole_blocks(ffmpeg, path, (bits, rate, channels, 0))
            if os.path.exists(path):
                os.remove(path)

            expected, right = expected_outcome(packet_sizes, packet_size, refused, whole)
            failures += not right
            print("24-bit %d Hz 5.1 in %s of %d-byte packets: %s; expected %s: %s"
                  % (rate, FORMAT_NAMES[output_format], packet_size, outcome, expected,
                     "ok" if right else "WRONG"))
    return failures


def pes_timestamp(value):
    """The five bytes of a PES header's PTS, in 90 kHz ticks."""
    return bytes([0x21 | ((value >> 30) & 7) << 1, (value >> 22) & 0xFF, ((value >> 14) & 0xFE) | 1,
                  (value >> 7) & 0xFF, ((value << 1) & 0xFE) | 1])


def pack_header(scr):
    """An MPEG-2 pack header with the given SCR and a mux rate of 10.08 Mb/s."""
    return bytes([0, 0, 1, 0xBA, 0x44 | ((scr >> 30) & 7) << 3 | ((scr >> 28) & 3),
                  (scr >> 20) & 0xFF, ((scr >> 15) & 0x1F) << 3 | 4 | ((scr >> 13) & 3),
                  (scr >> 5) & 0xFF, (scr & 0x1F) << 3 | 4, 1, 0x01, 0x89, 0xC3, 0xF8])


def write_dvd_pcm(path, bits, channels):
    """Writes to path a VOB of silent 48 kHz DVD PCM alone, as FFmpeg's encoder writes only 1, 2, 6
    and 8 channels. Each pack holds one packet of whole blocks with the PTS of its first sample, and
    each SCR is 0.2 s before that PTS."""
    block = block_size(bits, channels)
    size = 2000 - 2000 % block
    total = 48000 * SECONDS * channels * bits // 8
    total -= total % block
    with open(path, "wb") as out:
        for offset in range(0, total, size):
            pts = 45000 + offset * 8 * 90000 // (channels * bits * 48000)
            data = (b"\x81\x80\x05" + pes_timestamp(pts)
                    + bytes([0xA0, 1, 0, 4, 0, (bits - 16) // 4 << 6 | (channels - 1), 0x80])
                    + bytes(min(size, total - offset)))
            out.write(pack_header(pts - 18000) + b"\x00\x00\x01\xBD"
                      + len(data).to_bytes(2, "big") + data)
        out.write(b"\x00\x00\x01\xB9")


def first_sample_due(packets, bits, rate, channels):
    """When the first sample is due, in 90 kHz ticks: the first stamped packet's PTS, less the
    samples in front of the frame at its pointer. A first packet too small for a block holds no
    sample, so no frame starts in it and it carries no PTS."""
    offset = 0
    for packet in packets:
        if packet.pts is not None:
            return packet.pts - (offset + packet.pointer - 4) * 8 // (bits * channels) * 90000 / rate
        offset += len(packet.payload)
    raise ValueError("no LPCM packet carries a PTS")


def late_packets(vob, bits, rate, channels):
    """The LPCM packets whose pack starts to arrive after the first block that the packet
    completes is due, and the smallest margin in ms by which a pack starts before that, negative for
    a late one. Each sample is due one sample period after the one before."""
    block = block_size(bits, channels)
    packets = lpcm_packets(vob)
    start = first_sample_due(packets, bits, rate, channels)
    offset = late = 0
    margin = float("inf")
    for packet in packets:
        first_end = (offset // block + 1) * block
        if first_end <= offset + len(packet.payload):
            due = start + (first_end - block) * 8 // (bits * channels) * 90000 / rate
            late += packet.scr > due
            margin = min(margin, due - packet.scr)
        offset += len(packet.payload)
    return late, len(packets), margin / 90


def check_arrivals(ffmpeg, directory):
    """Every block of DVD PCM that FFmpeg writes alone into a VOB starts to arrive, by the SCR of
    its pack, before its first sample is due: 24-bit 5.1 that FFmpeg encodes, and 24-bit 7-channel
    DVD PCM that FFmpeg copies, in each of ARRIVAL_PACKET_SIZES. Each copied packet holds 1932
    bytes, so many of the packets that FFmpeg writes start none of them and carry no PTS, which
    leaves room for more samples."""
    raw = os.path.join(directory, "arrivals.s32le")
    with open(raw, "wb") as out:
        out.write(counted(24, 6, 48000 * SECONDS))
    copied = os.path.join(directory, "arrivals-7-channels.vob")
    write_dvd_pcm(copied, 24, 7)
    vob = os.path.join(directory, "arrivals.vob")
    sources = [("5.1", 6, ["-f", "s32le", "-ar", "48000", "-ch_layout", "5.1", "-i", raw,
                           "-c:a", "pcm_dvd", "-sample_fmt", "s32"]),
               ("7 channels, copied,", 7, ["-i", copied, "-c:a", "copy"])]
    failures = 0
    for packet_size in ARRIVAL_PACKET_SIZES:
        for name, channels, arguments in sources:
            run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"] + arguments
                + ["-packetsize", str(packet_size), "-f", "vob", vob])
            late, total, margin = late_packets(vob, 24, 48000, channels)
            failures += late != 0
            print("24-bit 48000 Hz %s alone in a VOB of %d-byte packets: %d of %d LPCM packets"
                  " start after a block they complete is due, smallest margin %+.2f ms: %s"
                  % (name, packet_size, late, total, margin, "ok" if not late else "WRONG"))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ffmpeg", required=True)
    arguments = parser.parse_args()
    failures = 0
    with tempfile.TemporaryDirectory() as directory:
        for shape in SHAPES:
            failures += check_shape(arguments.ffmpeg, directory, shape)
        for shape in VARYING_DYNAMIC_RANGE_SHAPES:
            failures += check_varying_dynamic_range(arguments.ffmpeg, directory, shape)
        for shape in START_BEFORE_ZERO_SHAPES:
            failures += check_start_before_zero(arguments.ffmpeg, directory, shape)
        failures += check_small_packets(arguments.ffmpeg, directory)
        failures += check_arrivals(arguments.ffmpeg, directory)
    print("%d checks failed" % failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
