#!/usr/bin/env python3
"""Checks the colours a filter writes when it converts Dolby Vision profile 5 to BT.2020 PQ.

    python3 streamarr/tests/dolby_vision_profile5_colour.py --ffmpeg PATH --filter NAME
                                                            [--cpuflags FLAGS] [--keep DIR]

Everything is synthetic. The picture is a frame of flat patches, each one base-layer code triple
(I, P, T): a grey ramp, rings of colours 16 to 260 codes around neutral, and bright colours whose
L'M'S' goes above the PQ range. FFmpeg's libx265 encodes it losslessly, and this script writes a
profile 5 RPU itself and appends it to the access unit. It checks seven RPUs on the same picture.
Two reshape P and T with one piece each: a polynomial, or MMR, which maps each of them from the
base-layer I, P and T together. The others split P and T into pieces at pivots:

    two polynomial pieces
    three polynomial pieces, two of them of second order
    eight first-order pieces, the most an RPU carries, with I in eight pieces too
    two MMR pieces of different orders
    an MMR piece, a polynomial piece, then another MMR piece

P and T never share a pivot code, and every piece reshapes at least MIN_PATCHES_PER_PIECE
checked patches.
Inside a flat patch, chroma siting and interpolation cannot change the result, so every output
code is compared with the value the profile 5 definition gives, computed here in double precision:

    1. s = code / 1023 for I, P and T (the base layer is full range)
    2. reshape each with its piece: the last piece whose lower pivot the component's own
       base-layer code reaches. A polynomial maps that component's s, and MMR maps the
       base-layer I, P and T. Clamp to 0..1
    3. L'M'S' = ycc_to_rgb_matrix x (I, P - 0.5, T - 0.5)
    4. LMS = PQ EOTF(L'M'S'), RGB = lms2rgb x rgb_to_lms_matrix x LMS, R'G'B' = PQ OETF(RGB)
    5. Y' = 0.2627 R' + 0.6780 G' + 0.0593 B', Cb = (B' - Y') / 1.8814, Cr = (R' - Y') / 1.4746
    6. Y = round(64 + 876 Y'), Cb = round(512 + 896 Cb), Cr = round(512 + 896 Cr)

A grey patch must be within GREY_TOLERANCE codes of that value and a coloured patch within
COLOUR_TOLERANCE. Over all checked samples, the mean of written minus exact code must be within
MAX_BIAS of zero for each component: a filter that rounds lands near 0, one that truncates near
-0.5, which the per-patch tolerance alone lets through. Colours within 1 % of the gamut boundary are not checked. Above the PQ range no
standard gives the value: L'M'S' above 1.0 still takes the EOTF formula, and tonemapx clips linear
RGB to 0..1 before the OETF where tonemap_cuda does not. So the bright colours whose L'M'S' exceeds
1.0, all of whose linear channels are above 200 nits, are checked for tonemapx only, with linear RGB
clipped to 0..1 in step 4. tonemap_cuda runs its
direct kernels (tradeoff=0): its 3D lookup table interpolates and is not held to these tolerances.
--cpuflags passes FFmpeg's -cpuflags to the filtering run, which picks the code path of a CPU
filter: 0 for plain C, sse4.2 for that path on a CPU that also has AVX2.
Exits 0 when every patch is within tolerance, 1 when one is not, 2 when the fixture is unusable.
"""
import argparse
import array
import collections
import math
import os
import subprocess
import sys
import tempfile

PATCH = 16
COLUMNS, ROWS = 24, 39
WIDTH, HEIGHT = COLUMNS * PATCH, ROWS * PATCH
GREY_TOLERANCE = 1
COLOUR_TOLERANCE = 1
MAX_BIAS = 0.25
MIN_PATCHES_PER_PIECE = 20

COEFFICIENT_BITS = 23
# A reshaping piece is a polynomial of the component's own value s: coefficients on 1, s and, for
# second order, s^2. Or it is MMR of the base-layer I, P and T: a constant, then for each order
# n the coefficients on I^n, P^n, T^n, (IP)^n, (IT)^n, (PT)^n and (IPT)^n.
Polynomial = collections.namedtuple("Polynomial", "coefficients")
Mmr = collections.namedtuple("Mmr", "constant orders")
# A component's curve: pivots in base-layer codes from 0 to 1023, and one piece between each two.
Curve = collections.namedtuple("Curve", "pivots pieces")


def one_piece(piece):
    return Curve((0, 1023), (piece,))


LUMA = one_piece(Polynomial((0.01, 0.85, 0.14)))
CHROMA = one_piece(Polynomial((0.03, 0.94)))
POLYNOMIAL, MMR = "polynomial", "MMR"
TWO_POLYNOMIALS, THREE_POLYNOMIALS, EIGHT_POLYNOMIALS, TWO_MMR, MIXED = (
    "two-piece polynomial chroma", "three-piece polynomial chroma", "eight-piece polynomial", "two-piece MMR chroma",
    "mixed MMR and polynomial chroma")
# name -> the curves of I, P and T. All but the eight-piece RPU reshape I with one second-order piece.
RPUS = {
    POLYNOMIAL: (LUMA, CHROMA, CHROMA),
    # P + 0.3 (I - 0.5) (P - 0.5), and T + 0.3 (I - 0.5) (T - 0.5) + 0.2 (P - 0.5) (T - 0.5).
    MMR: (LUMA, one_piece(Mmr(0.075, ((-0.15, 0.85, 0, 0.3, 0, 0, 0),))),
          one_piece(Mmr(0.125, ((-0.15, -0.1, 0.75, 0, 0.3, 0.2, 0),)))),
    # P: slope 0.80 below code 470 and 1.05 from it. T: slope 0.92 below code 540 and 0.70 from it.
    # Each curve is continuous at its pivot within 0.2 codes, and neutral stays neutral.
    TWO_POLYNOMIALS: (LUMA, Curve((0, 470, 1023), (Polynomial((0.09, 0.80)), Polynomial((-0.025, 1.05)))),
                      Curve((0, 540, 1023), (Polynomial((0.04, 0.92)), Polynomial((0.156, 0.70))))),
    # P: a second-order piece below code 400, a line through neutral up to code 620, and another
    # second-order piece from there; T the same with pivots at codes 430 and 590. Each piece meets
    # the next within 0.2 codes at the pivot between them, with another slope.
    THREE_POLYNOMIALS: (LUMA,
                        Curve((0, 400, 620, 1023), (Polynomial((0.1208, 0.6045, 0.25)), Polynomial((0.02, 0.96)),
                                                    Polynomial((-0.175, 1.4636, -0.30)))),
                        Curve((0, 430, 590, 1023), (Polynomial((-0.092, 1.3181, -0.20)), Polynomial((0.04, 0.92)),
                                                    Polynomial((0.2378, 0.404, 0.30))))),
    # I follows the chords of LUMA's curve between its pivots. P and T pass through neutral, and
    # their slope, from 0.80 to 1.18, changes at every pivot. Each curve is continuous at its pivots
    # within 0.1 codes.
    EIGHT_POLYNOMIALS: (
        Curve((0, 100, 220, 350, 470, 585, 715, 845, 1023),
              tuple(map(Polynomial, ((0.01, 0.8637), (0.0071, 0.8938), (-0.0003, 0.928), (-0.012, 0.9622),
                                     (-0.0268, 0.9944), (-0.046, 1.0279), (-0.0708, 1.0635), (-0.1056, 1.1056))))),
        Curve((0, 378, 418, 453, 506, 537, 582, 626, 1023),
              tuple(map(Polynomial, ((0.08, 0.80), (0.0504, 0.88), (0.0218, 0.95), (-0.0004, 1.00),
                                     (-0.03, 1.06), (0.0172, 0.97), (0.057, 0.90), (0.0938, 0.84))))),
        Curve((0, 350, 400, 441, 471, 507, 553, 620, 1023),
              tuple(map(Polynomial, ((-0.0281, 1.10), (-0.0007, 1.02), (0.0306, 0.94), (0.0565, 0.88),
                                     (0.0196, 0.96), (-0.02, 1.04), (-0.0633, 1.12), (-0.0996, 1.18)))))),
    # P: below code 540, P + 0.3 (I - 0.5) (P - 0.5) + 0.4 (P - 0.5)^2, of second order; from it,
    # P + 0.2 (P - 0.5) (T - 0.5) - 0.25 (I - 0.5) (P - 0.5). T: below code 470, the MMR RPU's T;
    # from it, T + 0.6 (T - 0.5)^3 - 0.2 (I - 0.5) (T - 0.5), of third order.
    TWO_MMR: (LUMA,
              Curve((0, 540, 1023), (Mmr(0.175, ((-0.15, 0.45, 0, 0.3, 0, 0, 0), (0, 0.4, 0, 0, 0, 0, 0))),
                                     Mmr(-0.0125, ((0.125, 1.025, -0.1, -0.25, 0, 0.2, 0),)))),
              Curve((0, 470, 1023), (Mmr(0.125, ((-0.15, -0.1, 0.75, 0, 0.3, 0.2, 0),)),
                                     Mmr(-0.125, ((0.1, 0, 1.55, 0, -0.2, 0, 0), (0, 0, -0.9, 0, 0, 0, 0),
                                                  (0, 0, 0.6, 0, 0, 0, 0)))))),
    # P: below code 440, the MMR RPU's P; up to code 555, 0.5 + (P - 0.5) + 1.41 (P - 0.5)^2;
    # from there, the second piece of the two-piece MMR RPU's P. T: below code 470,
    # T + 0.3 (I - 0.5) (T - 0.5) + 0.3 (T - 0.5)^2, of second order; up to code 597,
    # 0.5 + (T - 0.5) + 1.45 (T - 0.5)^2; from there,
    # T + 0.2 (P - 0.5) (T - 0.5) - 0.2 (I - 0.5) (T - 0.5). The polynomial pieces have a negative
    # coefficient on s, chosen so that blending it with the third piece in float32, x + (y - x),
    # lands just below the row where that piece's MMR weights start.
    MIXED: (LUMA,
            Curve((0, 440, 555, 1023), (Mmr(0.075, ((-0.15, 0.85, 0, 0.3, 0, 0, 0),)),
                                        Polynomial((0.3525, -0.41, 1.41)),
                                        Mmr(-0.0125, ((0.125, 1.025, -0.1, -0.25, 0, 0.2, 0),)))),
            Curve((0, 470, 597, 1023), (Mmr(0.15, ((-0.15, 0, 0.55, 0, 0.3, 0, 0), (0, 0, 0.3, 0, 0, 0, 0))),
                                        Polynomial((0.3625, -0.45, 1.45)),
                                        Mmr(0.0, ((0.1, -0.1, 1.0, 0, -0.2, 0.2, 0),))))),
}

# The IPT-PQ-C2 colour space of profile 5, as the RPU carries it.
YCC_TO_RGB = (8192, 799, 1681, 8192, -933, 1091, 8192, 267, -5545)  # / 2^13
RGB_TO_LMS = (17081, -349, -349, -349, 17081, -349, -349, -349, 17081)  # / 2^14
# HPE LMS to BT.2020 RGB, the constant that vf_tonemapx.c and vf_tonemap_cuda.c hold.
LMS_TO_RGB = ((3.06441879, -2.16597676, 0.10155818),
              (-0.65612108, 1.78554118, -0.12943749),
              (0.01736321, -0.04725154, 1.03004253))

# name -> (device options, filter graph, whether the bright colours above the PQ range are checked)
FILTERS = {
    "tonemapx": ([], "format=yuv420p10le,"
                     "tonemapx=t=smpte2084:p=bt2020:m=bt2020:r=tv:format=yuv420p10le", True),
    "tonemap_cuda": (["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu"],
                     "format=p010le,hwupload,"
                     "tonemap_cuda=t=smpte2084:p=bt2020:m=bt2020:r=tv:tradeoff=0:format=p010le,"
                     "hwdownload,format=p010le", False),
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


def piece_index(curve, code):
    """The piece whose range holds a base-layer code: the last one whose lower pivot the code reaches."""
    return max(k for k in range(len(curve.pieces)) if k == 0 or code >= curve.pivots[k])


def reshape(curve, component, triple):
    """Reshapes one component of a base-layer triple with the piece that its own code picks."""
    piece = curve.pieces[piece_index(curve, triple[component])]
    i, p, t = (code / 1023 for code in triple)
    if isinstance(piece, Polynomial):
        own = (i, p, t)[component]
        value = sum(fixed(c) * own ** k for k, c in enumerate(piece.coefficients))
    else:
        terms = (i, p, t, i * p, i * t, p * t, i * p * t)
        value = fixed(piece.constant) + sum(fixed(c) * term ** order
                                            for order, coefficients in enumerate(piece.orders, 1)
                                            for c, term in zip(coefficients, terms))
    return min(max(value / (1 << COEFFICIENT_BITS), 0.0), 1.0)


def multiply(matrix, vector):
    return [sum(m * v for m, v in zip(row, vector)) for row in matrix]


IN_GAMUT, ABOVE_RANGE = "in gamut", "above the PQ range"


def exact_codes(triple, rpu=POLYNOMIAL):
    """Returns the exact output codes of a base-layer triple and its kind: IN_GAMUT, ABOVE_RANGE
    or None for a colour that is not checked."""
    reshaped = [reshape(curve, component, triple) for component, curve in enumerate(RPUS[rpu])]
    signal = [reshaped[0], reshaped[1] - 0.5, reshaped[2] - 0.5]
    nonlinear = [YCC_TO_RGB[k:k + 3] for k in (0, 3, 6)]
    lms_pq = [value / 8192 for value in multiply(nonlinear, signal)]
    crosstalk = [[value / 16384 for value in RGB_TO_LMS[k:k + 3]] for k in (0, 3, 6)]
    linear = multiply(LMS_TO_RGB, multiply(crosstalk, [pq_eotf(value) for value in lms_pq]))
    r, g, b = (pq_oetf(min(value, 1.0)) for value in linear)
    luma = 0.2627 * r + 0.6780 * g + 0.0593 * b
    codes = (64 + 876 * luma, 512 + 896 * (b - luma) / 1.8814, 512 + 896 * (r - luma) / 1.4746)
    kind = None
    # Near the gamut boundary one channel is the small difference of large terms, and single
    # precision cannot hold it. Those colours are left out.
    if (min(lms_pq) >= 0 and max(lms_pq) <= 1 and max(linear) <= 1
            and min(linear) >= 0.01 * max(linear)):
        kind = IN_GAMUT
    elif 1 < max(lms_pq) <= 1.375 and min(linear) >= 0.02:
        kind = ABOVE_RANGE
    return [round(code) for code in codes], kind


def ring(luma, radius, hues):
    return [(luma, round(512 + radius * math.cos(2 * math.pi * hue / hues)),
             round(512 + radius * math.sin(2 * math.pi * hue / hues))) for hue in range(hues)]


def patch_triples():
    triples = [(round(1023 * k / 95), 512, 512) for k in range(96)]
    for luma in (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896):
        for radius in (16, 48, 90):
            triples += ring(luma, radius, 8)
        for radius in (140, 200, 260):
            triples += ring(luma, radius, 16)
    triples += [triple for luma in (960, 992, 1023) for radius in (90, 140) for triple in ring(luma, radius, 16)
                if exact_codes(triple)[1] == ABOVE_RANGE]
    triples += [(512, 512, 512)] * (-len(triples) % COLUMNS)
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
        """Writes fixed(value), the coefficient the reference uses: a whole part, then a fraction."""
        code = fixed(value)
        self.se(code >> COEFFICIENT_BITS)
        self.put(code & ((1 << COEFFICIENT_BITS) - 1), COEFFICIENT_BITS)

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


def rpu_nal(rpu):
    """A profile 5 RPU in the order libavcodec/dovi_rpudec.c reads it, as an HEVC NAL unit of type 62."""
    curves = RPUS[rpu]
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
    for curve in curves:
        bits.ue(len(curve.pivots) - 2)  # num_pivots_minus2
        for previous, pivot in zip((0,) + curve.pivots, curve.pivots):
            bits.put(pivot - previous, 10)  # each pivot as its distance from the one before
    bits.ue(0)          # num_x_partitions_minus1
    bits.ue(0)          # num_y_partitions_minus1
    for piece in (piece for curve in curves for piece in curve.pieces):
        if isinstance(piece, Mmr):
            bits.ue(1)           # mapping_idc: MMR
            bits.put(len(piece.orders) - 1, 2)  # mmr_order_minus1
            bits.coefficient(piece.constant)
            for value in (value for coefficients in piece.orders for value in coefficients):
                bits.coefficient(value)
            continue
        bits.ue(0)      # mapping_idc: polynomial
        bits.ue(len(piece.coefficients) - 2)  # poly_order_minus1
        if len(piece.coefficients) == 2:
            bits.put(0, 1)       # linear_interp_flag
        for value in piece.coefficients:
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


def slug(rpu):
    return rpu.lower().replace(" ", "-")


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


def check(triples, planes, rpu, arguments):
    """Prints the patches out of tolerance. Returns how many were checked, how many failed and, per
    component and piece, how many checked patches that piece reshapes."""
    above_range = FILTERS[arguments.filter][2]
    checked, failures = 0, []
    difference, samples_checked = [0, 0, 0], [0, 0, 0]
    checked_per_piece = collections.Counter()
    largest = [0, 0, 0]
    for index, triple in enumerate(triples):
        expected, kind = exact_codes(triple, rpu)
        if kind != IN_GAMUT and not (kind == ABOVE_RANGE and above_range):
            continue
        checked += 1
        for component, curve in enumerate(RPUS[rpu]):
            checked_per_piece[component, piece_index(curve, triple[component])] += 1
        tolerance = GREY_TOLERANCE if triple[1:] == (512, 512) else COLOUR_TOLERANCE
        written = [interior(planes[component], index, component) for component in range(3)]
        for component, samples in enumerate(written):
            difference[component] += sum(samples) - expected[component] * len(samples)
            samples_checked[component] += len(samples)
        furthest = [max(samples, key=lambda sample: abs(sample - expected[component]))
                    for component, samples in enumerate(written)]
        errors = [abs(furthest[component] - expected[component]) for component in range(3)]
        largest = [max(pair) for pair in zip(largest, errors)]
        if any(errors[component] > tolerance for component in range(3)):
            failures.append((triple, expected, furthest))
    for triple, expected, furthest in failures[:12]:
        print("%s, I/P/T %4d/%4d/%4d: exact Y/Cb/Cr %4d/%4d/%4d, %s wrote %4d/%4d/%4d"
              % ((rpu,) + tuple(triple) + tuple(expected) + (arguments.filter,) + tuple(furthest)))
    print("%s, %s RPU: %d of %d checked patches are further from the exact code than allowed "
          "(grey %d, colour %d); largest error Y %d, Cb %d, Cr %d"
          % ((arguments.filter, rpu, len(failures), checked, GREY_TOLERANCE, COLOUR_TOLERANCE) + tuple(largest)))
    bias = [total / count for total, count in zip(difference, samples_checked)]
    biased = any(abs(value) > MAX_BIAS for value in bias)
    print("%s, %s RPU: mean of written minus exact code Y %+.3f, Cb %+.3f, Cr %+.3f (allowed %.2f): %s"
          % ((arguments.filter, rpu) + tuple(bias) + (MAX_BIAS, "BIASED" if biased else "ok")))
    return checked, len(failures) + biased, checked_per_piece


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--filter", required=True, choices=sorted(FILTERS))
    parser.add_argument("--cpuflags", help="FFmpeg's -cpuflags for the filtering run")
    parser.add_argument("--keep", help="write the fixture and the output to this directory")
    arguments = parser.parse_args()
    triples = patch_triples()
    device, graph, _ = FILTERS[arguments.filter]
    cpuflags = ["-cpuflags", arguments.cpuflags] if arguments.cpuflags is not None else []
    outputs = {}
    with tempfile.TemporaryDirectory() as temporary:
        folder = arguments.keep or temporary
        os.makedirs(folder, exist_ok=True)
        source, base = (os.path.join(folder, name) for name in ("patches.yuv", "base.hevc"))
        frame = picture(triples)
        with open(source, "wb") as file:
            file.write(frame)
        ffmpeg = [arguments.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        run(ffmpeg + ["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "%dx%d" % (WIDTH, HEIGHT),
                      "-color_range", "pc", "-i", source, "-frames:v", "1", "-c:v", "libx265",
                      "-x265-params", "lossless=1:range=full:log-level=error", "-f", "hevc", base])
        with open(base, "rb") as file:
            access_unit = file.read()
        for rpu in RPUS:
            stream = os.path.join(folder, "profile5-%s.hevc" % slug(rpu))
            with open(stream, "wb") as file:
                file.write(access_unit + rpu_nal(rpu))
            decoded = run(ffmpeg + ["-i", stream, "-frames:v", "1", "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"])
            if decoded != frame:
                sys.stderr.write("The decoded base layer differs from the patches, so the fixture is unusable.\n")
                sys.exit(2)
            output = run(ffmpeg + cpuflags + device + ["-i", stream, "-vf", graph, "-frames:v", "1",
                                                       "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"])
            if arguments.keep:
                with open(os.path.join(folder, "%s-%s.yuv" % (arguments.filter, slug(rpu))), "wb") as file:
                    file.write(output)
            if len(output) != len(frame):
                sys.stderr.write("%s wrote %d bytes where one frame is %d.\n"
                                 % (arguments.filter, len(output), len(frame)))
                sys.exit(2)
            outputs[rpu] = planes_of(output)
    failed = 0
    for rpu, planes in outputs.items():
        checked, failures, checked_per_piece = check(triples, planes, rpu, arguments)
        if checked < 300:
            sys.stderr.write("Only %d of %d patches are checked, so the fixture is unusable.\n"
                             % (checked, len(triples)))
            sys.exit(2)
        for component, curve in enumerate(RPUS[rpu]):
            for piece in range(len(curve.pieces)):
                if checked_per_piece[component, piece] < MIN_PATCHES_PER_PIECE:
                    sys.stderr.write("Piece %d of %s in the %s RPU reshapes only %d checked patches, "
                                     "so the fixture is unusable.\n"
                                     % (piece + 1, "IPT"[component], rpu, checked_per_piece[component, piece]))
                    sys.exit(2)
        failed += failures
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
