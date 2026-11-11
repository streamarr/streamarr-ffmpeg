#!/usr/bin/env python3
"""Checks the colours a filter writes when it converts Dolby Vision profile 5 to BT.2020 PQ.

    python3 streamarr/tests/dolby_vision_profile5_colour.py --ffmpeg PATH --filter NAME [--keep DIR]

Everything is synthetic. The picture is a frame of flat patches, each one base-layer code triple
(I, P, T): a grey ramp and three rings of colours around neutral. FFmpeg's libx265 encodes it
losslessly, and this script writes the profile 5 RPU itself and appends it to the access unit.
Inside a flat patch, chroma siting and interpolation cannot change the result, so every output
code is compared with the value the profile 5 definition gives, computed here in double precision:

    1. s = code / 1023 for I, P and T (the base layer is full range)
    2. reshape each with its polynomial, clamp to 0..1
    3. L'M'S' = ycc_to_rgb_matrix x (I, P - 0.5, T - 0.5)
    4. LMS = PQ EOTF(L'M'S'), RGB = lms2rgb x rgb_to_lms_matrix x LMS, R'G'B' = PQ OETF(RGB)
    5. Y' = 0.2627 R' + 0.6780 G' + 0.0593 B', Cb = (B' - Y') / 1.8814, Cr = (R' - Y') / 1.4746
    6. Y = round(64 + 876 Y'), Cb = round(512 + 896 Cb), Cr = round(512 + 896 Cr)

A grey patch must be within GREY_TOLERANCE codes of that value and a coloured patch within
COLOUR_TOLERANCE. Exits 0 when every patch is, 1 when one is not, 2 when the fixture is unusable.
"""
import argparse
import array
import math
import os
import subprocess
import sys
import tempfile

PATCH = 16
COLUMNS, ROWS = 24, 15
WIDTH, HEIGHT = COLUMNS * PATCH, ROWS * PATCH
GREY_TOLERANCE = 1
COLOUR_TOLERANCE = 2

# The RPU's reshaping curves: one second-order piece for I, one first-order piece for P and T.
COEFFICIENT_BITS = 23
LUMA_CURVE = (0.01, 0.85, 0.14)
CHROMA_CURVE = (0.03, 0.94)

# The IPT-PQ-C2 colour space of profile 5, as the RPU carries it.
YCC_TO_RGB = (8192, 799, 1681, 8192, -933, 1091, 8192, 267, -5545)  # / 2^13
RGB_TO_LMS = (17081, -349, -349, -349, 17081, -349, -349, -349, 17081)  # / 2^14
# HPE LMS to BT.2020 RGB, the constant that vf_tonemapx.c and vf_tonemap_cuda.c hold.
LMS_TO_RGB = ((3.06441879, -2.16597676, 0.10155818),
              (-0.65612108, 1.78554118, -0.12943749),
              (0.01736321, -0.04725154, 1.03004253))

FILTERS = {
    "tonemap_cuda": (["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu"],
                     "format=p010le,hwupload,"
                     "tonemap_cuda=t=smpte2084:p=bt2020:m=bt2020:r=tv:format=p010le,"
                     "hwdownload,format=p010le"),
}

PQ_M1, PQ_M2 = 2610 / 16384, 2523 / 4096 * 128
PQ_C1, PQ_C2, PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def fixed(value):
    return round(value * (1 << COEFFICIENT_BITS))


def pq_eotf(signal):
    power = max(signal, 0.0) ** (1 / PQ_M2)
    return (max(power - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * power)) ** (1 / PQ_M1)


def pq_oetf(linear):
    power = max(linear, 0.0) ** PQ_M1
    return ((PQ_C1 + PQ_C2 * power) / (1 + PQ_C3 * power)) ** PQ_M2


def reshape(curve, signal):
    value = sum(fixed(c) / (1 << COEFFICIENT_BITS) * signal ** k for k, c in enumerate(curve))
    return min(max(value, 0.0), 1.0)


def multiply(matrix, vector):
    return [sum(m * v for m, v in zip(row, vector)) for row in matrix]


def exact_codes(triple):
    """Returns the exact output codes of a base-layer triple and whether the colour is in gamut."""
    i, p, t = (code / 1023 for code in triple)
    signal = [reshape(LUMA_CURVE, i), reshape(CHROMA_CURVE, p) - 0.5, reshape(CHROMA_CURVE, t) - 0.5]
    nonlinear = [YCC_TO_RGB[k:k + 3] for k in (0, 3, 6)]
    lms_pq = [value / 8192 for value in multiply(nonlinear, signal)]
    crosstalk = [[value / 16384 for value in RGB_TO_LMS[k:k + 3]] for k in (0, 3, 6)]
    linear = multiply(LMS_TO_RGB, multiply(crosstalk, [pq_eotf(value) for value in lms_pq]))
    r, g, b = (pq_oetf(value) for value in linear)
    luma = 0.2627 * r + 0.6780 * g + 0.0593 * b
    codes = (64 + 876 * luma, 512 + 896 * (b - luma) / 1.8814, 512 + 896 * (r - luma) / 1.4746)
    # Near the gamut boundary one channel is the small difference of large terms, and single
    # precision cannot hold it. Those colours are left out.
    in_gamut = (min(lms_pq) >= 0 and max(lms_pq) <= 1 and max(linear) <= 1
                and min(linear) >= 0.01 * max(linear))
    return [round(code) for code in codes], in_gamut


def patch_triples():
    triples = [(round(1023 * k / 95), 512, 512) for k in range(96)]
    for luma in (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896):
        for radius in (16, 48, 90):
            for hue in range(8):
                angle = 2 * math.pi * hue / 8
                triples.append((luma, round(512 + radius * math.cos(angle)),
                                round(512 + radius * math.sin(angle))))
    assert len(triples) == COLUMNS * ROWS
    return triples


def picture(triples):
    luma = array.array("H")
    for row in range(ROWS):
        line = array.array("H")
        for column in range(COLUMNS):
            line.extend([triples[row * COLUMNS + column][0]] * PATCH)
        for _ in range(PATCH):
            luma.extend(line)
    planes = [luma]
    for component in (1, 2):
        plane = array.array("H")
        for row in range(ROWS):
            line = array.array("H")
            for column in range(COLUMNS):
                line.extend([triples[row * COLUMNS + column][component]] * (PATCH // 2))
            for _ in range(PATCH // 2):
                plane.extend(line)
        planes.append(plane)
    if sys.byteorder == "big":
        for plane in planes:
            plane.byteswap()
    return b"".join(plane.tobytes() for plane in planes)


class Bits:
    def __init__(self):
        self.bits = []

    def put(self, value, count):
        self.bits.extend((value >> shift) & 1 for shift in range(count - 1, -1, -1))

    def ue(self, value):
        self.put(value + 1, 2 * (value + 1).bit_length() - 1)

    def se(self, value):
        self.ue(2 * value - 1 if value > 0 else -2 * value)

    def coefficient(self, value):
        whole = math.floor(value)
        self.se(whole)
        self.put(round((value - whole) * (1 << COEFFICIENT_BITS)), COEFFICIENT_BITS)

    def to_bytes(self):
        padded = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, padded[k:k + 8])), 2) for k in range(0, len(padded), 8))


def crc32_mpeg2(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def rpu_nal():
    """A profile 5 RPU in the order libavcodec/dovi_rpudec.c reads it, as an HEVC NAL unit of type 62."""
    bits = Bits()
    bits.put(2, 6)      # rpu_type
    bits.put(18, 11)    # rpu_format
    bits.put(0, 4)      # vdr_rpu_profile
    bits.put(0, 4)      # vdr_rpu_level
    bits.put(1, 1)      # vdr_seq_info_present_flag
    bits.put(0, 1)      # chroma_resampling_explicit_filter_flag
    bits.put(0, 2)      # coefficient_data_type: fixed point
    bits.ue(COEFFICIENT_BITS)
    bits.put(1, 2)      # vdr_rpu_normalized_idc
    bits.put(1, 1)      # bl_video_full_range_flag
    bits.ue(2)          # bl_bit_depth_minus8
    bits.ue(2)          # el_bit_depth_minus8
    bits.ue(4)          # vdr_bit_depth_minus8
    bits.put(0, 1)      # spatial_resampling_filter_flag
    bits.put(0, 3)      # reserved
    bits.put(0, 1)      # el_spatial_resampling_filter_flag
    bits.put(1, 1)      # disable_residual_flag
    bits.put(1, 1)      # vdr_dm_metadata_present_flag
    bits.put(0, 1)      # use_prev_vdr_rpu_flag
    bits.ue(0)          # vdr_rpu_id
    bits.ue(0)          # mapping_color_space
    bits.ue(0)          # mapping_chroma_format_idc
    for _ in range(3):
        bits.ue(0)      # num_pivots_minus2
        bits.put(0, 10)
        bits.put(1023, 10)
    bits.ue(0)          # num_x_partitions_minus1
    bits.ue(0)          # num_y_partitions_minus1
    for curve in (LUMA_CURVE, CHROMA_CURVE, CHROMA_CURVE):
        bits.ue(0)      # mapping_idc: polynomial
        bits.ue(len(curve) - 2)  # poly_order_minus1
        if len(curve) == 2:
            bits.put(0, 1)       # linear_interp_flag
        for value in curve:
            bits.coefficient(value)
    bits.ue(0)          # affected_dm_metadata_id
    bits.ue(0)          # current_dm_metadata_id
    bits.ue(1)          # scene_refresh_flag
    for value in YCC_TO_RGB:
        bits.put(value & 0xFFFF, 16)
    for value in (0, 1 << 27, 1 << 27):  # ycc_to_rgb_offset: 0, 0.5, 0.5
        bits.put(value, 32)
    for value in RGB_TO_LMS:
        bits.put(value & 0xFFFF, 16)
    bits.put(65535, 16)  # signal_eotf
    bits.put(0, 16)
    bits.put(0, 16)
    bits.put(0, 32)
    bits.put(12, 5)     # signal_bit_depth
    bits.put(2, 2)      # signal_color_space
    bits.put(0, 2)      # signal_chroma_format
    bits.put(1, 2)      # signal_full_range_flag
    bits.put(62, 12)    # source_min_PQ
    bits.put(3696, 12)  # source_max_PQ
    bits.put(42, 10)    # source_diagonal
    bits.ue(0)          # num_ext_blocks
    body = bits.to_bytes()
    payload = b"\x19" + body + crc32_mpeg2(body).to_bytes(4, "big") + b"\x80"
    escaped = bytearray()
    zeros = 0
    for byte in payload:
        if zeros >= 2 and byte <= 3:
            escaped.append(3)
            zeros = 0
        escaped.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return b"\x00\x00\x00\x01\x7c\x01" + bytes(escaped)


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        sys.exit(2)
    return result.stdout


def planes_of(raw):
    samples = array.array("H")
    samples.frombytes(raw[:WIDTH * HEIGHT * 3])
    if sys.byteorder == "big":
        samples.byteswap()
    luma_count, chroma_count = WIDTH * HEIGHT, WIDTH * HEIGHT // 4
    return (samples[:luma_count], samples[luma_count:luma_count + chroma_count],
            samples[luma_count + chroma_count:luma_count + 2 * chroma_count])


def interior(plane, index, component):
    """The samples of a patch that are at least a quarter of a patch away from its edge."""
    size = PATCH if component == 0 else PATCH // 2
    stride = WIDTH if component == 0 else WIDTH // 2
    top, left = (index // COLUMNS) * size, (index % COLUMNS) * size
    margin = size // 4
    return [plane[(top + y) * stride + left + x]
            for y in range(margin, size - margin) for x in range(margin, size - margin)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--filter", required=True, choices=sorted(FILTERS))
    parser.add_argument("--keep", help="write the fixture and the output to this directory")
    arguments = parser.parse_args()
    triples = patch_triples()
    with tempfile.TemporaryDirectory() as temporary:
        folder = arguments.keep or temporary
        os.makedirs(folder, exist_ok=True)
        source, base, stream = (os.path.join(folder, name) for name in ("patches.yuv", "base.hevc", "profile5.hevc"))
        frame = picture(triples)
        with open(source, "wb") as file:
            file.write(frame)
        ffmpeg = [arguments.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        run(ffmpeg + ["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "%dx%d" % (WIDTH, HEIGHT),
                      "-color_range", "pc", "-i", source, "-frames:v", "1", "-c:v", "libx265",
                      "-x265-params", "lossless=1:range=full:log-level=error", "-f", "hevc", base])
        with open(base, "rb") as file:
            access_unit = file.read()
        with open(stream, "wb") as file:
            file.write(access_unit + rpu_nal())
        decoded = run(ffmpeg + ["-i", stream, "-frames:v", "1", "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"])
        if decoded != frame:
            sys.stderr.write("The decoded base layer differs from the patches, so the fixture is unusable.\n")
            sys.exit(2)
        device, graph = FILTERS[arguments.filter]
        output = run(ffmpeg + device + ["-i", stream, "-vf", graph, "-frames:v", "1",
                                        "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"])
        if arguments.keep:
            with open(os.path.join(folder, arguments.filter + ".yuv"), "wb") as file:
                file.write(output)
    if len(output) != len(frame):
        sys.stderr.write("%s wrote %d bytes where one frame is %d.\n" % (arguments.filter, len(output), len(frame)))
        sys.exit(2)
    planes = planes_of(output)
    checked, failures = 0, []
    for index, triple in enumerate(triples):
        expected, in_gamut = exact_codes(triple)
        if not in_gamut:
            continue
        checked += 1
        tolerance = GREY_TOLERANCE if triple[1:] == (512, 512) else COLOUR_TOLERANCE
        written = [interior(planes[component], index, component) for component in range(3)]
        furthest = [max(samples, key=lambda sample: abs(sample - expected[component]))
                    for component, samples in enumerate(written)]
        if any(abs(furthest[component] - expected[component]) > tolerance for component in range(3)):
            failures.append((triple, expected, furthest))
    if checked < 300:
        sys.stderr.write("Only %d of %d patches are in gamut, so the fixture is unusable.\n" % (checked, len(triples)))
        sys.exit(2)
    for triple, expected, furthest in failures[:12]:
        print("I/P/T %4d/%4d/%4d: exact Y/Cb/Cr %4d/%4d/%4d, %s wrote %4d/%4d/%4d"
              % (tuple(triple) + tuple(expected) + (arguments.filter,) + tuple(furthest)))
    print("%s: %d of %d in-gamut patches are further from the exact code than allowed "
          "(grey %d, colour %d)" % (arguments.filter, len(failures), checked, GREY_TOLERANCE, COLOUR_TOLERANCE))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
