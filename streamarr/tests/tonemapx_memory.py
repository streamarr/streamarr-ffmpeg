#!/usr/bin/env python3
"""Checks that tonemapx frees what it allocates.

    python3 streamarr/tests/tonemapx_memory.py --ffmpeg PATH [--cpuflags FLAGS]
    python3 streamarr/tests/tonemapx_memory.py --ffmpeg PATH --leak-sanitizer [--cpuflags FLAGS]

Everything is synthetic. FFmpeg's libx265 encodes one small black frame, and the stream is that
access unit again and again, each copy followed by the polynomial profile 5 RPU of the colour
check (dolby_vision_profile5_colour.py). It runs through the two tonemapx chains of the Streamarr
transcode worker: the reshape of Dolby Vision profile 5 to BT.2020 PQ, and the tone map to SDR.
Nothing is encoded after the filter.

Without --leak-sanitizer the check runs with any build, and measures. The filter may keep a fixed
amount of memory however many frames it converts. The growth of a chain is the peak resident
memory of a run of LONG_FRAMES frames less that of a run of SHORT_FRAMES, for each further frame.
A chain with tonemapx may grow at most BYTES_PER_FRAME faster than the same decoding without the
filter. One copy of the parsed RPU, struct FFDOVIMetadataRemap, is 2,728 bytes, so a filter that
keeps one per frame grows more than five times faster than allowed. Every run decodes and filters
on one thread each, which keeps the memory that does not depend on the frame count equal.

With --leak-sanitizer the check is exact, and needs a Linux build with AddressSanitizer and
libx265 (configure --toolchain=gcc-asan or clang-asan): Apple's clang has no LeakSanitizer. It
runs SANITIZED_FRAMES frames with RPUs, and as many without, through both chains, and
LeakSanitizer must find every allocation freed. The release archives are built without the
sanitizer, so this mode runs only where someone builds with it.

--cpuflags passes FFmpeg's -cpuflags to the filtering runs: 0 for plain C, sse4.2 for that path on
a CPU that also has AVX2.
Exits 0 when memory is freed, 1 when it is not, 2 when the check cannot run: the fixture is
unusable, FFmpeg fails, or decoding alone grows by more than BYTES_PER_FRAME.
"""
import argparse
import os
import subprocess
import sys
import tempfile

from dolby_vision_profile5_colour import POLYNOMIAL, rpu_nal, run

CHAINS = {
    "reshape to BT.2020 PQ": "format=yuv420p10le,"
                             "tonemapx=t=smpte2084:p=bt2020:m=bt2020:r=tv:format=yuv420p10le",
    "tone map to SDR": "tonemapx=tonemap=mobius:peak=100:desat=0:t=bt709:m=bt709:p=bt709:r=tv:"
                       "format=yuv420p",
}
DECODING_ALONE = "format=yuv420p10le"
SHORT_FRAMES, LONG_FRAMES = 2000, 42000
BYTES_PER_FRAME = 512
SANITIZED_FRAMES = 48
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
    log = result.stderr.decode(errors="replace")
    if result.returncode != 0:
        sys.stderr.write(log)
        sys.stderr.write("FFmpeg exited with %d.\n" % result.returncode)
        sys.exit(2)
    metadata = [line for line in log.splitlines() if "Dolby Vision Metadata" in line]
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


def growth_per_frame(arguments, stream, graph):
    short, long = (peak_resident_bytes(filtering(arguments, stream, graph, frames))
                   for frames in (SHORT_FRAMES, LONG_FRAMES))
    return short, long, (long - short) / (LONG_FRAMES - SHORT_FRAMES)


def growth_failures(arguments, stream):
    _, _, decoding = growth_per_frame(arguments, stream, DECODING_ALONE)
    if decoding > BYTES_PER_FRAME:
        sys.stderr.write("Decoding alone grows by %.0f bytes a frame, so the check cannot tell "
                         "what tonemapx keeps.\n" % decoding)
        sys.exit(2)
    failures = 0
    for name, graph in CHAINS.items():
        short, long, per_frame = growth_per_frame(arguments, stream, graph)
        grew = per_frame - decoding > BYTES_PER_FRAME
        failures += grew
        print("%s: peak resident memory %.1f MB after %d frames and %.1f MB after %d, %.0f bytes "
              "for each further frame, %.0f without tonemapx (%d more allowed)%s"
              % (name, short / 2**20, SHORT_FRAMES, long / 2**20, LONG_FRAMES, per_frame, decoding,
                 BYTES_PER_FRAME, ": tonemapx keeps memory for every frame" if grew else ""))
    return failures


def require_sanitizer(arguments):
    configuration = run([arguments.ffmpeg, "-hide_banner", "-buildconf"]).decode()
    if "asan" not in configuration and "-fsanitize=address" not in configuration:
        sys.stderr.write("--leak-sanitizer needs a build configured with --toolchain=gcc-asan "
                         "or clang-asan.\n")
        sys.exit(2)


def sanitizer_failures(arguments, streams):
    options = [os.environ["ASAN_OPTIONS"]] if os.environ.get("ASAN_OPTIONS") else []
    environment = dict(os.environ, ASAN_OPTIONS=":".join(options + ["detect_leaks=1"]))
    failures = 0
    for source, stream in streams.items():
        for name, graph in CHAINS.items():
            result = subprocess.run(filtering(arguments, stream, graph, SANITIZED_FRAMES),
                                    env=environment, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            report = result.stderr.decode(errors="replace").splitlines()
            leaks = [index for index, line in enumerate(report) if " leak of " in line]
            if result.returncode != 0 and not leaks:
                sys.stderr.write("\n".join(report))
                sys.stderr.write("\nFFmpeg exited with %d.\n" % result.returncode)
                sys.exit(2)
            failures += bool(leaks)
            if not leaks:
                print("%s, %s: every allocation was freed" % (source, name))
            for index in leaks:
                # The innermost frame of the allocation's stack that is in the filter's own file.
                site = next((frame.split(" in ", 1)[1] for frame in report[index + 1:index + 8]
                             if "vf_tonemapx" in frame), "outside vf_tonemapx.c")
                print("%s, %s: %s %s" % (source, name, report[index].strip().rstrip(":"), site))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--cpuflags", help="FFmpeg's -cpuflags for the filtering runs")
    parser.add_argument("--leak-sanitizer", action="store_true",
                        help="use LeakSanitizer; needs a build with AddressSanitizer")
    arguments = parser.parse_args()
    if arguments.leak_sanitizer:
        require_sanitizer(arguments)
    with tempfile.TemporaryDirectory() as folder:
        base, profile5, plain = (os.path.join(folder, name)
                                 for name in ("base.hevc", "profile5.hevc", "plain.hevc"))
        run([arguments.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=64x64:r=24,format=yuv420p10le",
             "-frames:v", "1", "-color_range", "pc", "-c:v", "libx265",
             "-x265-params", "lossless=1:range=full:info=0:log-level=error", "-f", "hevc", base])
        with open(base, "rb") as file:
            access_unit = file.read()
        frames = SANITIZED_FRAMES if arguments.leak_sanitizer else LONG_FRAMES
        write_stream(profile5, access_unit + rpu_nal(POLYNOMIAL), frames)
        require_reshaped_frames(arguments, profile5)
        if arguments.leak_sanitizer:
            write_stream(plain, access_unit, frames)
            failures = sanitizer_failures(arguments, {"with RPUs": profile5, "without RPUs": plain})
        else:
            failures = growth_failures(arguments, profile5)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
