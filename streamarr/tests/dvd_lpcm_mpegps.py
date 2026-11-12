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

It also writes 24-bit 5.1, whose block is 72 bytes, alone into packets too small for a block. A
VOB packet of 96 bytes has room for 60 after the pack header that every MPEG-2 packet carries, a
PES header with a PTS and the LPCM header. An MPEG-1 system stream packet of 88 bytes has room for
70 without a pack header. No block fits, so FFmpeg must refuse those packet sizes. A 128-byte
packet has room for one block even with a pack header, a PTS and a DTS, so FFmpeg must write it.
Every size between is refused or written in whole blocks, and FFmpeg must stop: a muxer that
cannot write a block writes empty packets forever.

Exits 0 when every check passes, 1 when one fails, 2 when FFmpeg fails.
"""
import argparse
import array
import os
import resource
import subprocess
import sys
import tempfile

SECONDS = 6
SEEKS = (1.3, 2.6, 3.9, 5.2)
# bits, rate and channels of every shape FFmpeg's encoder writes under DVD-Video's 9.8 Mb/s
SHAPES = [(16, 48000, 1), (16, 48000, 2), (16, 48000, 6), (16, 96000, 6),
          (24, 48000, 1), (24, 48000, 2), (24, 48000, 6), (24, 96000, 1), (24, 96000, 2)]
LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1"}
SMALL_PACKETS = {"vob": range(96, 129), "mpeg": range(88, 129)}
FORMAT_NAMES = {"vob": "a VOB", "mpeg": "an MPEG-1 system stream"}
SMALL_PACKET_SAMPLES = 24000
# Half a second of 24-bit 5.1 fills about 0.7 MB of these packets. A muxer that never drains its
# queue writes past this limit within a second, and the limit stops it.
OUTPUT_LIMIT = 16 << 20


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
    bits, rate, channels = shape
    raw = os.path.join(directory, "b%d-r%d-c%d.s32le" % shape)
    vob = os.path.join(directory, "b%d-r%d-c%d.vob" % shape)
    with open(raw, "wb") as out:
        out.write(counted(bits, channels, rate * SECONDS))
    run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=%d" % SECONDS,
         "-f", "s32le", "-ar", str(rate), "-ch_layout", LAYOUTS[channels], "-i", raw,
         "-c:v", "mpeg2video", "-g", "12", "-bf", "0",
         "-c:a", "pcm_dvd", "-sample_fmt", "s16" if bits == 16 else "s32", "-f", "vob", vob])
    return vob


def pes_body(data, position):
    """Where the data of the PES packet at position starts, after an MPEG-2 or MPEG-1 header."""
    if data[position + 6] & 0xC0 == 0x80:
        return position + 9 + data[position + 8]
    body = position + 6
    while data[body] == 0xFF:
        body += 1
    if data[body] & 0xC0 == 0x40:
        body += 2
    timestamps = data[body] & 0xF0
    if timestamps == 0x20:
        return body + 5
    if timestamps == 0x30:
        return body + 10
    return body + 1


def lpcm_payloads(vob):
    """The (frame count, first access unit pointer, payload) of every LPCM packet, in order, in
    an MPEG-2 program stream or an MPEG-1 system stream."""
    with open(vob, "rb") as source:
        data = source.read()
    payloads = []
    position = 0
    while position + 4 <= len(data):
        if data[position:position + 3] != b"\x00\x00\x01":
            raise ValueError("no start code at byte %d" % position)
        stream_id = data[position + 3]
        if stream_id == 0xB9:
            break
        if stream_id == 0xBA:
            pack_header = 12
            if data[position + 4] & 0xC0 == 0x40:
                pack_header = 14 + (data[position + 13] & 7)
            position += pack_header
            continue
        end = position + 6 + int.from_bytes(data[position + 4:position + 6], "big")
        if stream_id == 0xBD:
            body = pes_body(data, position)
            if 0xA0 <= data[body] <= 0xAF:
                pointer = int.from_bytes(data[body + 2:body + 4], "big")
                payloads.append((data[body + 1], pointer, data[body + 7:end]))
        position = end
    return payloads


def misaligned_payloads(vob, bits, channels):
    block = block_size(bits, channels)
    offset = misaligned = 0
    payloads = lpcm_payloads(vob)
    for _, _, payload in payloads:
        misaligned += offset % block != 0
        offset += len(payload)
    return misaligned, len(payloads)


def decoded_rows(ffmpeg, vob, bits, channels, seek):
    """Each decoded row as (source sample index, or None for a row that names none)."""
    seek_arguments = ["-ss", str(seek)] if seek is not None else []
    pcm = run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"] + seek_arguments
              + ["-i", vob, "-map", "0:a:0", "-c:a", "pcm_s32le", "-f", "s32le", "-"])
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
    return rows, period


def rows_out_of_sequence(rows, period):
    """The rows that hold no source sample or do not follow the row before."""
    wrong = 0
    previous = None
    for row in rows:
        if row is None or (previous is not None and row != (previous + 1) % period):
            wrong += 1
        previous = row
    return wrong


def check_shape(ffmpeg, directory, shape):
    bits, rate, channels = shape
    name = "%d-bit %d Hz %s" % (bits, rate, LAYOUTS[channels])
    vob = write_source(ffmpeg, directory, shape)
    failures = 0

    misaligned, total = misaligned_payloads(vob, bits, channels)
    failures += misaligned != 0
    print("%s: %d of %d LPCM payloads start inside a %d-byte block: %s"
          % (name, misaligned, total, block_size(bits, channels), "ok" if not misaligned else "WRONG"))

    rows, period = decoded_rows(ffmpeg, vob, bits, channels, None)
    expected = rate * SECONDS
    whole = rows == [n % period for n in range(expected)]
    failures += not whole
    print("%s from the start: %d rows for %d source samples, %d out of sequence: %s"
          % (name, len(rows), expected, rows_out_of_sequence(rows, period), "ok" if whole else "WRONG"))

    for seek in SEEKS:
        rows, period = decoded_rows(ffmpeg, vob, bits, channels, seek)
        wrong = rows_out_of_sequence(rows, period)
        right = rows and wrong == 0
        failures += not right
        print("%s after a seek to %.1f s: %d of %d rows hold no source sample or do not follow"
              " the one before: %s" % (name, seek, wrong, len(rows), "ok" if right else "WRONG"))
    return failures


def written_in_whole_blocks(ffmpeg, vob, bits, channels):
    """Whether every payload starts on a block and the decode from the start holds every source
    sample in order, and a description of what the file holds."""
    misaligned, total = misaligned_payloads(vob, bits, channels)
    rows, period = decoded_rows(ffmpeg, vob, bits, channels, None)
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
                whole, outcome = written_in_whole_blocks(ffmpeg, path, bits, channels)
            if os.path.exists(path):
                os.remove(path)

            expected, right = expected_outcome(packet_sizes, packet_size, refused, whole)
            failures += not right
            print("24-bit %d Hz 5.1 in %s of %d-byte packets: %s; expected %s: %s"
                  % (rate, FORMAT_NAMES[output_format], packet_size, outcome, expected,
                     "ok" if right else "WRONG"))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ffmpeg", required=True)
    arguments = parser.parse_args()
    failures = 0
    with tempfile.TemporaryDirectory() as directory:
        for shape in SHAPES:
            failures += check_shape(arguments.ffmpeg, directory, shape)
        failures += check_small_packets(arguments.ffmpeg, directory)
    print("%d checks failed" % failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
