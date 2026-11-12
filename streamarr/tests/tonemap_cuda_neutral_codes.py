#!/usr/bin/env python3
"""Checks the codes tonemap_cuda writes for neutral black, white and grey.

    python3 streamarr/tests/tonemap_cuda_neutral_codes.py --ffmpeg PATH

The inputs are flat BT.2020 PQ frames with neutral chroma, written here code for code. The filter
clips them to BT.709 (tonemap=clip, tonemap_mode=max), which keeps neutral input neutral, leaves
black at zero and takes PQ peak white to exactly one. BT.709 and BT.2100 table 9 then fix the
output codes without any reference implementation:

    narrow range   black Y 16 (64 in 10 bits), white Y 235 (940), Cb and Cr 128 (512)
    full range     black Y 0, white Y 255 (1023), Cb and Cr 128 (512)

Grey must keep Cb and Cr neutral. Its luma is not fixed, and 8-bit output dithers it, so it may
cover two adjacent codes. Every case runs twice: with the direct kernels (tradeoff=0) and with the
3D lookup table that the filter picks on small GPUs (tradeoff=1). The table interpolates between
nodes, and black, white and neutral chroma are nodes, so it must write these codes as well.
Needs an NVIDIA GPU. Exits 0 when every code is right, 1 when one is not, 2 when FFmpeg fails.
"""
import argparse
import array
import subprocess
import sys

WIDTH, HEIGHT = 256, 144
# output format -> (array type, bits to drop, bit depth)
FORMATS = {"nv12": ("B", 0, 8), "p010le": ("H", 6, 10)}
# name -> (input range, input black, input white, input grey)
INPUTS = {"tv": (64, 940, 502), "pc": (0, 1023, 512)}


def expected(depth, output_range):
    """Black luma, white luma and neutral chroma of BT.2100 table 9 at this depth."""
    if output_range == "tv":
        return 16 << (depth - 8), 235 << (depth - 8), 128 << (depth - 8)
    return 0, (1 << depth) - 1, 1 << (depth - 1)


def codes(ffmpeg, case):
    input_range, input_luma, output_range, pixel_format, tradeoff = case
    frame = array.array("H", [input_luma] * (WIDTH * HEIGHT) + [512] * (WIDTH * HEIGHT // 2))
    if sys.byteorder == "big":
        frame.byteswap()
    graph = ("format=p010le,hwupload,tonemap_cuda=tonemap=clip:tonemap_mode=max:t=bt709:m=bt709:"
             "p=bt709:r=%s:tradeoff=%d:format=%s,hwdownload,format=%s"
             % (output_range, tradeoff, pixel_format, pixel_format))
    result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                             "-init_hw_device", "cuda=cu", "-filter_hw_device", "cu",
                             "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "%dx%d" % (WIDTH, HEIGHT),
                             "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                             "-colorspace", "bt2020nc", "-color_range", input_range, "-i", "-",
                             "-vf", graph, "-frames:v", "1", "-f", "rawvideo", "-"],
                            input=frame.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    kind, shift, _ = FORMATS[pixel_format]
    samples = array.array(kind)
    if result.returncode != 0 or len(result.stdout) != WIDTH * HEIGHT * 3 // 2 * samples.itemsize:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        sys.stderr.write("FFmpeg failed or wrote %d bytes for %s\n" % (len(result.stdout), (case,)))
        sys.exit(2)
    samples.frombytes(result.stdout)
    if sys.byteorder == "big" and kind == "H":
        samples.byteswap()
    luma = sorted({sample >> shift for sample in samples[:WIDTH * HEIGHT]})
    chroma = sorted({sample >> shift for sample in samples[WIDTH * HEIGHT:]})
    return luma, chroma


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ffmpeg", required=True)
    arguments = parser.parse_args()
    wrong = total = 0
    for tradeoff in (0, 1):
        for input_range, output_range in (("tv", "tv"), ("tv", "pc"), ("pc", "tv")):
            for pixel_format, (_, _, depth) in sorted(FORMATS.items()):
                black, white, neutral = expected(depth, output_range)
                for name, input_luma, luma_wanted in zip(("black", "white", "grey"), INPUTS[input_range],
                                                         ([black], [white], None)):
                    luma, chroma = codes(arguments.ffmpeg,
                                         (input_range, input_luma, output_range, pixel_format, tradeoff))
                    luma_right = luma == luma_wanted if luma_wanted else luma[-1] - luma[0] <= 1
                    right = luma_right and chroma == [neutral]
                    wrong += not right
                    total += 1
                    print("tradeoff=%d %s to %s %-6s %-5s Y %s (expected %s), Cb and Cr %s (expected [%d]): %s"
                          % (tradeoff, input_range, output_range, pixel_format, name, luma,
                             luma_wanted or "one code or two adjacent", chroma, neutral,
                             "ok" if right else "WRONG"))
    print("%d of %d cases are wrong" % (wrong, total))
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
