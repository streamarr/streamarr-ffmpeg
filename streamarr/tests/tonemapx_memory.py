#!/usr/bin/env python3
"""Checks that tonemapx's memory does not grow with the number of frames it converts.

    python3 streamarr/tests/tonemapx_memory.py --ffmpeg PATH [--cpuflags FLAGS]

Everything is synthetic. FFmpeg's libx265 encodes one small black frame, and the stream is that
access unit again and again, each copy followed by the profile 5 RPU of the colour check
(dolby_vision_profile5_colour.py). It runs through the two tonemapx chains of the Streamarr
transcode worker: the reshape of Dolby Vision profile 5 to BT.2020 PQ, and the tone map to SDR.
Nothing is encoded after the filter.

The filter may keep a fixed amount of memory however many frames it converts. So the peak
resident memory of a run of LONG_FRAMES frames may exceed that of a run of SHORT_FRAMES by at most
BYTES_PER_FRAME for each further frame. One copy of the parsed RPU, struct FFDOVIMetadataRemap, is
2,728 bytes, so a filter that keeps one per frame grows more than five times faster than allowed.
Both runs decode and filter on one thread each, which keeps the memory that does not depend on the
frame count equal.

--cpuflags passes FFmpeg's -cpuflags to the filtering runs: 0 for plain C, sse4.2 for that path on
a CPU that also has AVX2.
Exits 0 when memory stays level, 1 when it grows, 2 when the check cannot run.
"""
import argparse
import os
import subprocess
import sys
import tempfile

from dolby_vision_profile5_colour import rpu_nal, run

CHAINS = {
    "reshape to BT.2020 PQ": "format=yuv420p10le,"
                             "tonemapx=t=smpte2084:p=bt2020:m=bt2020:r=tv:format=yuv420p10le",
    "tone map to SDR": "tonemapx=tonemap=mobius:peak=100:desat=0:t=bt709:m=bt709:p=bt709:r=tv:"
                       "format=yuv420p",
}
SHORT_FRAMES, LONG_FRAMES = 2000, 42000
BYTES_PER_FRAME = 512
# ru_maxrss is in bytes on macOS and in kilobytes on Linux.
PEAK_UNIT = 1 if sys.platform == "darwin" else 1024


def write_stream(path, access_unit, frames):
    # In pieces: on Linux a child's peak resident memory starts at its parent's size when it was
    # forked, so this process must stay smaller than the FFmpeg runs it measures.
    with open(path, "wb") as file:
        for _ in range(frames):
            file.write(access_unit)


def require_reshaped_frames(arguments, stream):
    """Stops unless the decoder hands the filter the metadata of an RPU that tonemapx reshapes."""
    result = subprocess.run([arguments.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info",
                             "-i", stream, "-frames:v", "1", "-vf", "showinfo", "-f", "null", "-"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    metadata = [line for line in result.stderr.decode(errors="replace").splitlines()
                if "Dolby Vision Metadata" in line]
    if not any("vdr_rpu_profile=0;" in line and "disable_residual_flag=1" in line
               for line in metadata):
        sys.stderr.write("The decoded frames carry no profile 5 metadata, so the fixture is "
                         "unusable.\n")
        sys.exit(2)


def filtering(arguments, stream, graph, frames):
    cpuflags = ["-cpuflags", arguments.cpuflags] if arguments.cpuflags is not None else []
    return ([arguments.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"] + cpuflags
            + ["-threads", "1", "-filter_threads", "1", "-i", stream, "-vf", graph,
               "-frames:v", str(frames), "-f", "null", "-"])


def peak_resident_bytes(command):
    """Runs the command and returns the peak resident memory of that one process."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
        if process.returncode != 0:
            errors.seek(0)
            sys.stderr.write(errors.read().decode(errors="replace"))
            sys.stderr.write("FFmpeg exited with %d.\n" % process.returncode)
            sys.exit(2)
    return usage.ru_maxrss * PEAK_UNIT


def growth_failures(arguments, stream):
    failures = 0
    for name, graph in CHAINS.items():
        short, long = (peak_resident_bytes(filtering(arguments, stream, graph, frames))
                       for frames in (SHORT_FRAMES, LONG_FRAMES))
        per_frame = (long - short) / (LONG_FRAMES - SHORT_FRAMES)
        grew = per_frame > BYTES_PER_FRAME
        failures += grew
        print("%s: peak resident memory %.1f MB after %d frames and %.1f MB after %d, "
              "%.0f bytes for each further frame (%d allowed)%s"
              % (name, short / 2**20, SHORT_FRAMES, long / 2**20, LONG_FRAMES, per_frame,
                 BYTES_PER_FRAME, ": the filter keeps memory for every frame" if grew else ""))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--cpuflags", help="FFmpeg's -cpuflags for the filtering runs")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        base, profile5 = (os.path.join(folder, name) for name in ("base.hevc", "profile5.hevc"))
        run([arguments.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=64x64:r=24,format=yuv420p10le",
             "-frames:v", "1", "-color_range", "pc", "-c:v", "libx265",
             "-x265-params", "lossless=1:range=full:info=0:log-level=error", "-f", "hevc", base])
        with open(base, "rb") as file:
            access_unit = file.read()
        write_stream(profile5, access_unit + rpu_nal(), LONG_FRAMES)
        require_reshaped_frames(arguments, profile5)
        failures = growth_failures(arguments, profile5)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
