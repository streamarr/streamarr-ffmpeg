#!/usr/bin/env python3
"""Checks the codes tonemap_cuda writes for black, white and grey when it tone maps PQ to BT.709.

    python3 streamarr/tests/tonemap_cuda_neutral_codes.py --ffmpeg PATH

The inputs are flat limited-range BT.2020 PQ frames with neutral chroma, written here code for
code. BT.709 fixes the answer without any reference implementation: black is Y 16 with Cb and Cr
128 in 8 bits (64 and 512 in 10 bits), PQ peak white tone maps to Y 235 (940), and a grey input
keeps Cb and Cr at 128 (512).
Needs an NVIDIA GPU. Exits 0 when every code is right, 1 when one is not, 2 when FFmpeg fails.
"""
import argparse
import array
import subprocess
import sys

WIDTH, HEIGHT = 256, 144
# output format -> (bytes per sample, bits to drop, black luma, white luma, neutral chroma)
FORMATS = {"nv12": ("B", 0, 16, 235, 128), "p010le": ("H", 6, 64, 940, 512)}


def codes(ffmpeg, input_luma, pixel_format):
    frame = array.array("H", [input_luma] * (WIDTH * HEIGHT) + [512] * (WIDTH * HEIGHT // 2))
    if sys.byteorder == "big":
        frame.byteswap()
    graph = ("format=p010le,hwupload,tonemap_cuda=tonemap=bt2390:t=bt709:m=bt709:p=bt709:format=%s,"
             "hwdownload,format=%s" % (pixel_format, pixel_format))
    result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                             "-init_hw_device", "cuda=cu", "-filter_hw_device", "cu",
                             "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "%dx%d" % (WIDTH, HEIGHT),
                             "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                             "-colorspace", "bt2020nc", "-color_range", "tv", "-i", "-",
                             "-vf", graph, "-frames:v", "1", "-f", "rawvideo", "-"],
                            input=frame.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        sys.exit(2)
    kind, shift = FORMATS[pixel_format][:2]
    samples = array.array(kind)
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
    wrong = 0
    for pixel_format, (_, _, black, white, neutral) in sorted(FORMATS.items()):
        for colour, input_luma, expected_luma in (("black", 64, [black]), ("white", 940, [white]), ("grey", 502, None)):
            luma, chroma = codes(arguments.ffmpeg, input_luma, pixel_format)
            right = chroma == [neutral] and (expected_luma is None or luma == expected_luma)
            wrong += not right
            print("%-6s %-5s Y %s (expected %s), Cb and Cr %s (expected [%d]): %s"
                  % (pixel_format, colour, luma, expected_luma or "any one code", chroma, neutral,
                     "ok" if right else "WRONG"))
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
