#!/usr/bin/env python3
"""Checks that agapfill writes a continuous audio timeline and keeps every real sample in place.

    python3 streamarr/tests/agapfill.py --ffmpeg PATH [--only NAME ...] [--list]

Everything is synthetic. The source is a stereo counter that FFmpeg's aevalsrc writes: sample n is
(n mod 30011 - 15005, n div 30011 + 1). Every sample names its own index, and no sample is (0, 0),
so each output sample is either one source sample or silence. Filters ahead of agapfill cut holes
in the source (aselect) or change its timestamps (asetpts). ashowinfo after agapfill reports the
timestamp and length of every frame, and FFmpeg writes the samples as raw PCM.

Each check states the timeline it expects: the first timestamp, then runs of source samples and
of silence. The expectations come from the rules the filter documents, not from its output:

    1. Output is continuous. Each frame begins where the previous frame ends.
    2. A forward step between two frames that is longer than `threshold` (2 ms) is a gap. The
       filter writes silence up to the later frame's own timestamp.
    3. Any other step, an overlap and a backward step included, writes no silence and drops no
       sample. The frame follows the previous frame directly.
    4. `start` puts silence from that time up to the first frame.
    5. Silence comes one frame of `packet_size` samples at a time, so memory does not grow with
       the gap, and a gap longer than `max_gap` is an error.
    6. When ffmpeg builds a new filter graph mid-stream, the new graph's filter continues the
       output of the old one, so rules 1 to 4 hold across the change. The old one first outputs
       every frame it has.

Exits 0 when every check passes, 1 when one fails, 2 when a fixture is unusable.
"""
import argparse
import array
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import threading

RATE = 48000
PERIOD, HALF = 30011, 15005
DEFAULT_PACKET = 4096
CHUNK = 1 << 16
RSS_UNIT = 1 if sys.platform == "darwin" else 1024
ADDRESS_SPACE = 4 << 30
# No check needs more than a few seconds. The limit only ends a run that would never end.
TIME_LIMIT = 120
# A run that never ends logs a frame of silence at a time, about 30 MB a second, into a temporary
# file that may sit in memory. No check logs more than 16 MB, and FFmpeg is stopped past this size.
FILE_SIZE = 256 << 20
SHOWINFO = re.compile(r"Parsed_ashowinfo\S* @ \S+ n:\d+ pts:(-?\d+|NOPTS) pts_time:(\S+) fmt:(\S+) "
                      r"channels:\d+ chlayout:\S+ rate:\d+ nb_samples:(\d+)")


class Unusable(Exception):
    """A fixture did not come out as the check needs it."""


class Counter:
    """The stereo counter source in one sample format."""

    def __init__(self, sample_format):
        self.sample_format = sample_format
        if sample_format == "s16":
            self.typecode, self.gain, self.full_scale = "h", 1, 1 << 15
            self.codec, self.raw = "pcm_s16le", "s16le"
        else:
            # 24 significant bits in a 32-bit sample: the low byte is zero.
            self.typecode, self.gain, self.full_scale = "i", 257, 1 << 23
            self.codec, self.raw = "pcm_s32le", "s32le"
        self.scale = 1 if sample_format == "s16" else 256
        self.item = array.array(self.typecode).itemsize
        self.width = 2 * self.item
        self._samples = array.array(self.typecode)

    def lavfi(self, count, rate):
        left = "(mod(n,%d)-%d)*%d/%d" % (PERIOD, HALF, self.gain, self.full_scale)
        right = "(floor(n/%d)+1)*%d/%d" % (PERIOD, self.gain, self.full_scale)
        return ("aevalsrc=exprs='%s|%s':s=%d,atrim=end_sample=%d,aformat=sample_fmts=%s"
                % (left, right, rate, count, self.sample_format))

    def samples(self, first, count):
        """The bytes of source samples first .. first + count - 1."""
        have = len(self._samples) // 2
        if first + count > have:
            factor = self.gain * self.scale
            more = array.array(self.typecode, bytes(2 * (first + count - have) * self.item))
            more[0::2] = array.array(self.typecode, [(n % PERIOD - HALF) * factor
                                                     for n in range(have, first + count)])
            more[1::2] = array.array(self.typecode, [(n // PERIOD + 1) * factor
                                                     for n in range(have, first + count)])
            self._samples.extend(more)
        part = self._samples[2 * first:2 * (first + count)]
        if sys.byteorder != "little":
            part.byteswap()
        return part.tobytes()

    def name(self, sample):
        """Says what one stereo output sample is."""
        values = array.array(self.typecode, sample)
        if sys.byteorder != "little":
            values.byteswap()
        left, right = values
        if (left, right) == (0, 0):
            return "silence"
        factor = self.gain * self.scale
        if left % factor or right % factor or right < factor or abs(left // factor) > HALF:
            return "the values (%d, %d), which are not a source sample" % (left, right)
        return "source sample %d" % ((right // factor - 1) * PERIOD + left // factor + HALF)


COUNTERS = {"s16": Counter("s16"), "s32": Counter("s32")}


def real(first, count):
    return ("real", first, count)


def silence(count):
    return ("silence", 0, count)


# A small interpreter that runs FFmpeg and reports the peak memory of its child. The figure is an
# upper bound: it also counts the interpreter's own memory up to the moment FFmpeg replaces the
# child, and a lean FFmpeg build stays below that. Run from this script itself, the figure would
# count the source samples this script holds.
MEASURE = """
import resource, subprocess, sys
own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
code = subprocess.call(sys.argv[1:])
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
sys.stderr.write("\\nmeasured: own %d peak %d\\n" % (own, peak))
sys.exit(code if code >= 0 else 1)
"""
MEASURED = re.compile(r"^measured: own (\d+) peak (\d+)$", re.MULTILINE)


def limit_memory():
    """Keeps a filter that queues a whole gap, or a run that never ends, from taking the machine's
    memory. The limit of the address space is Linux only."""
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE, ADDRESS_SPACE))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE, FILE_SIZE))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class Finished:
    def __init__(self, code, log, result):
        self.code, self.log, self.result = code, log, result

    def peak(self):
        """An upper bound of FFmpeg's peak resident memory in bytes, from a measured run."""
        match = MEASURED.search(self.log)
        if not match:
            raise Unusable("FFmpeg's peak memory could not be measured")
        return int(match.group(2)) * RSS_UNIT


def execute(command, consume=None, timeout=TIME_LIMIT, measure=False):
    """Runs FFmpeg. `consume` reads its standard output. Returns its status, log and result."""
    if measure:
        command = [sys.executable, "-c", MEASURE] + command
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stderr=log,
                                   stdout=subprocess.PIPE if consume else subprocess.DEVNULL,
                                   preexec_fn=limit_memory)
        expired = []

        def stop():
            expired.append(True)
            process.kill()

        timer = threading.Timer(timeout, stop)
        timer.start()
        try:
            result = consume(process.stdout) if consume else None
            if consume:
                process.stdout.close()
            process.wait()
        finally:
            timer.cancel()
        log.seek(0)
        text = log.read().decode("utf-8", "replace")
    if expired:
        text += "\nThe command did not end within %d s and was stopped.\n" % timeout
    if process.returncode == -signal.SIGXFSZ:
        text += "\nThe command wrote %d MB to a file and was stopped.\n" % (FILE_SIZE >> 20)
    return Finished(process.returncode, text, result)


def byte_count(stream):
    """Reads a stream to its end, one chunk at a time, and returns its length."""
    count = 0
    for part in iter(lambda: stream.read(CHUNK), b""):
        count += len(part)
    return count


def last_lines(log, count=3):
    lines = [line for line in log.splitlines() if line.strip() and "Parsed_ashowinfo" not in line]
    return " | ".join(lines[-count:])


def frames_of(log):
    """The (timestamp, length, format, time in seconds) of every frame ashowinfo saw, in order."""
    frames = []
    for match in SHOWINFO.finditer(log):
        pts = None if match.group(1) == "NOPTS" else int(match.group(1))
        seconds = None if pts is None else float(match.group(2))
        frames.append((pts, int(match.group(4)), match.group(3), seconds))
    return frames


def read_fully(stream, size):
    parts = []
    while size:
        part = stream.read(size)
        if not part:
            break
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def compare_samples(stream, first, runs, counter, rate):
    """Reads raw PCM and returns what differs from the expected runs."""
    problems, position, width = [], 0, counter.width
    expected_total = sum(count for _, _, count in runs)
    for kind, source_first, count in runs:
        done = 0
        while done < count:
            want = min(count - done, CHUNK)
            got = read_fully(stream, want * width)
            have = len(got) // width
            if kind == "real":
                expected = counter.samples(source_first + done, have)
            else:
                expected = bytes(have * width)
            if got[:have * width] != expected and not problems:
                index = next(i for i in range(have)
                             if got[i * width:(i + 1) * width] != expected[i * width:(i + 1) * width])
                at = first + position + index
                problems.append("the sample at %d (%.6f s) is %s, expected %s"
                                % (at, at / rate, counter.name(got[index * width:(index + 1) * width]),
                                   counter.name(expected[index * width:(index + 1) * width])))
            position += have
            done += have
            if have < want:
                problems.append("the output ends after %d samples, expected %d"
                                % (position, expected_total))
                return problems
    extra = 0
    while True:
        part = stream.read(CHUNK * width)
        if not part:
            break
        extra += len(part) // width
    if extra:
        problems.append("the output holds %d samples, expected %d"
                        % (expected_total + extra, expected_total))
    return problems


def check_frames(frames, first, runs, packet_size, rate):
    """Checks the timestamps of the frames against the expected timeline."""
    problems = []
    total = sum(count for _, _, count in runs)
    if not frames:
        return [] if not total else ["no frame came out of the filter"]
    if frames[0][0] != first:
        problems.append("the first frame is stamped %s, expected %d" % (frames[0][0], first))
    # ashowinfo prints the time with six significant digits.
    pts, _, _, seconds = frames[-1]
    if pts is not None and abs(seconds - pts / rate) > 1e-5 * max(1, abs(pts / rate)):
        problems.append("the frame stamped %d is at %s s, expected %.6f s: the output time base is "
                        "not one sample" % (pts, seconds, pts / rate))
    breaks = [(frames[i - 1][0], frames[i - 1][1], frames[i][0]) for i in range(1, len(frames))
              if frames[i - 1][0] is None or frames[i][0] != frames[i - 1][0] + frames[i - 1][1]]
    if breaks:
        pts, length, following = breaks[0]
        end = "nowhere" if pts is None else "%d (%.6f s)" % (pts + length, (pts + length) / rate)
        problems.append("the output timeline breaks %d times: first at a frame stamped %s, "
                        "which follows a frame that ends at %s" % (len(breaks), following, end))
        return problems
    position, silences = first, []
    for kind, _, count in runs:
        if kind == "silence":
            silences.append((position, position + count))
        position += count
    long_frames = [(pts, length) for pts, length, _, _ in frames if length > packet_size
                   and any(begin <= pts and pts + length <= end for begin, end in silences)]
    if long_frames:
        problems.append("%d frames of silence are longer than %d samples: the first has %d"
                        % (len(long_frames), packet_size, long_frames[0][1]))
    return problems


def filter_command(ffmpeg, source, chain, counter):
    return [ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-f", "lavfi", "-i", source,
            "-af", chain + ",ashowinfo", "-c:a", counter.codec, "-f", counter.raw, "-"]


def timeline_problems(ffmpeg, chain, first, runs, *, count, sample_format="s16", rate=RATE,
                      packet_size=DEFAULT_PACKET):
    """Runs the counter through `chain` and compares the output with the expected timeline."""
    counter = COUNTERS[sample_format]
    command = filter_command(ffmpeg, counter.lavfi(count, rate), chain, counter)
    done = execute(command, lambda out: compare_samples(out, first, runs, counter, rate))
    if done.code:
        return ["FFmpeg ended with status %d: %s" % (done.code, last_lines(done.log))]
    return check_frames(frames_of(done.log), first, runs, packet_size, rate) + done.result


# ffmpeg's exit status for "invalid argument": the low byte of AVERROR(EINVAL), which is -22. A
# crash or a sanitizer report after the filter's error line ends with another status.
REFUSED = 234


def refusal_problems(ffmpeg, chain, words, *, count):
    """Expects FFmpeg to refuse `chain` with an error line that holds `words`."""
    counter = COUNTERS["s16"]
    done = execute(filter_command(ffmpeg, counter.lavfi(count, RATE), chain, counter), byte_count)
    if done.code != REFUSED:
        return ["FFmpeg ended with status %d after %d bytes, expected status %d: %s"
                % (done.code, done.result, REFUSED, last_lines(done.log))]
    errors = [line for line in done.log.splitlines() if "agapfill" in line and words in line]
    if not errors:
        return ["FFmpeg failed without an agapfill error that says '%s': %s"
                % (words, last_lines(done.log))]
    return []


# --- Shapes: filters ahead of agapfill that make the timestamps under test. ---------------------

def frames(length):
    return "asetnsamples=n=%d:p=0" % length


def drop(*ranges):
    """Removes whole frames, counted from zero, so their time holds no audio."""
    return "aselect='not(%s)'" % "+".join("between(n,%d,%d)" % pair for pair in ranges)


def stamp(expression):
    """Rewrites timestamps. N is the number of samples before the frame, PTS its timestamp."""
    return "asetpts='%s'" % re.sub(r"\bN\b", "NB_CONSUMED_SAMPLES", expression)


def chain(*parts):
    return ",".join(parts)


SECOND = RATE
EIGHT_SECOND_GAP = chain(frames(4800), drop((60, 139)))
EIGHT_SECOND_TIMELINE = [real(0, 6 * SECOND), silence(8 * SECOND), real(14 * SECOND, 10 * SECOND)]
LATE_BY_8_SECONDS = chain(frames(4800), stamp("PTS+384000"))
TWO_HOUR_GAP = chain(frames(4800), stamp("PTS+345600000*gte(N,96000)"))

CHECKS = []


def check(name, title):
    def register(function):
        CHECKS.append((name, title, function))
        return function
    return register


def timeline_check(name, title, shape, options, first, runs, *, after=None, **settings):
    @check(name, title)
    def run(ffmpeg):
        fill = "agapfill" + ("=" + options if options else "")
        parts = [shape, fill] + ([after] if after else [])
        return timeline_problems(ffmpeg, chain(*parts), first, runs, **settings)


def refusal_check(name, title, shape, options, words, *, count):
    @check(name, title)
    def run(ffmpeg):
        return refusal_problems(ffmpeg, chain(shape, "agapfill=" + options), words, count=count)


# --- Gaps ---------------------------------------------------------------------------------------

timeline_check("gap", "fills a gap of 8 s with silence and leaves every sample at its own time",
               EIGHT_SECOND_GAP, "", 0, EIGHT_SECOND_TIMELINE, count=24 * SECOND)

timeline_check("gap-24-bit", "fills a gap in 24-bit audio without changing a sample",
               EIGHT_SECOND_GAP, "", 0, EIGHT_SECOND_TIMELINE, count=24 * SECOND,
               sample_format="s32")

timeline_check("gap-between-odd-frames", "fills a gap whose edges are on no frame grid",
               chain(frames(1000), drop((289, 672))), "", 0,
               [real(0, 289000), silence(384000), real(673000, 479000)], count=24 * SECOND)

timeline_check("two-gaps", "fills two gaps, each for its own length",
               chain(frames(4800), drop((60, 139), (180, 199))), "", 0,
               [real(0, 6 * SECOND), silence(8 * SECOND), real(14 * SECOND, 4 * SECOND),
                silence(2 * SECOND), real(20 * SECOND, 4 * SECOND)], count=24 * SECOND)

timeline_check("gap-of-30-ms", "fills a gap of 30 ms",
               chain(frames(480), drop((600, 602))), "", 0,
               [real(0, 288000), silence(1440), real(289440, 190560)], count=10 * SECOND)

# --- The threshold decides each step by itself --------------------------------------------------

timeline_check("step-of-1-ms", "writes no silence for a step of 1 ms",
               chain(frames(480), stamp("PTS+48*gte(N,288000)")), "", 0,
               [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("step-of-2-ms", "writes no silence for a step of exactly 2 ms, the threshold",
               chain(frames(480), stamp("PTS+96*gte(N,288000)")), "", 0,
               [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("step-above-2-ms", "fills a step one sample longer than 2 ms",
               chain(frames(480), stamp("PTS+97*gte(N,288000)")), "", 0,
               [real(0, 288000), silence(97), real(288000, 192000)], count=10 * SECOND)

timeline_check("step-of-88-samples-at-44100", "writes no silence for 88 samples at 44.1 kHz, "
               "which is below 2 ms", chain(frames(441), stamp("PTS+88*gte(N,264600)")), "", 0,
               [real(0, 441000)], count=441000, rate=44100)

timeline_check("step-of-89-samples-at-44100", "fills 89 samples at 44.1 kHz, which is above 2 ms",
               chain(frames(441), stamp("PTS+89*gte(N,264600)")), "", 0,
               [real(0, 264600), silence(89), real(264600, 176400)], count=441000, rate=44100)

timeline_check("threshold-option-step-at", "writes no silence for a step equal to threshold=0.01",
               chain(frames(480), stamp("PTS+480*gte(N,288000)")), "threshold=0.01", 0,
               [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("threshold-option-step-above", "fills a step one sample above threshold=0.01",
               chain(frames(480), stamp("PTS+481*gte(N,288000)")), "threshold=0.01", 0,
               [real(0, 288000), silence(481), real(288000, 192000)], count=10 * SECOND)

timeline_check("threshold-of-zero", "fills a step of one sample with threshold=0",
               chain(frames(480), stamp("PTS+gte(N,288000)")), "threshold=0", 0,
               [real(0, 288000), silence(1), real(288000, 192000)], count=10 * SECOND)

timeline_check("small-steps-do-not-add-up", "writes no silence for nine steps of 2 ms, 18 ms in all",
               chain(frames(4800), stamp("PTS+96*floor(N/48000)")), "", 0,
               [real(0, 10 * SECOND)], count=10 * SECOND)

# The five small steps before the gap leave the output 480 samples behind the timestamps. The fill
# covers them too, so the first sample after the gap is at its own time, 14 s plus 14 steps.
timeline_check("gap-after-small-steps", "fills a gap up to the next frame's own timestamp",
               chain(frames(4800), stamp("PTS+96*floor(N/48000)"), drop((60, 139))), "", 0,
               [real(0, 288000), silence(673344 - 288000), real(672000, 480000)],
               count=24 * SECOND)

# --- Real audio is never trimmed ----------------------------------------------------------------

timeline_check("overlap", "keeps every sample when timestamps overlap by 50 ms",
               chain(frames(480), stamp("PTS-2400*gte(N,288000)")), "", 0,
               [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("backward-step", "keeps every sample when timestamps go back by one hour",
               chain(frames(480), stamp("PTS-172800000*gte(N,288000)")), "", 0,
               [real(0, 10 * SECOND)], count=10 * SECOND)

# After the overlap the output is 2,400 samples ahead of the timestamps. The gap needs that much
# less silence to put the next frame at its own time.
timeline_check("gap-after-overlap", "fills a gap after an overlap only up to the next timestamp",
               chain(frames(4800), stamp("PTS-2400*gte(N,144000)"), drop((60, 139))), "", 0,
               [real(0, 288000), silence(384000 - 2400), real(672000, 480000)],
               count=24 * SECOND)

timeline_check("gap-shorter-than-overlap", "writes no silence and drops nothing when a gap of "
               "30 ms follows an overlap of 50 ms",
               chain(frames(480), stamp("PTS-2400*gte(N,144000)"), drop((600, 602))), "", 0,
               [real(0, 288000), real(289440, 190560)], count=10 * SECOND)

timeline_check("timestamps-run-fast", "inserts nothing when timestamps run 0.02 % fast",
               chain(frames(4800), stamp("floor(PTS*1.0002)")), "", 0,
               [real(0, 24 * SECOND)], count=24 * SECOND)

timeline_check("timestamps-run-slow", "drops nothing when timestamps run 0.02 % slow",
               chain(frames(4800), stamp("floor(PTS*0.9998)")), "", 0,
               [real(0, 24 * SECOND)], count=24 * SECOND)

# Timestamps step forward by 2 ms at 1 s, so the output is 96 samples behind them. The frames from
# 2 s to 4 s have no timestamp. The frame at 4 s continues the timestamps from before 2 s. Unless
# the frames without a timestamp count toward the end of the previous frame, it looks like a gap.
timeline_check("frames-without-timestamp", "places frames without a timestamp after the previous "
               "frame", chain(frames(4800),
                              stamp("if(between(N,96000,191999),NAN,PTS+96*gte(N,48000))")),
               "", 0, [real(0, 10 * SECOND)], count=10 * SECOND)

# --- Silence before audio that starts late ------------------------------------------------------

timeline_check("late-start-without-option", "writes no leading silence without the start option",
               LATE_BY_8_SECONDS, "", 384000, [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("late-start", "writes silence from start=0 up to audio that begins at 8 s",
               LATE_BY_8_SECONDS, "start=0", 0, [silence(384000), real(0, 10 * SECOND)],
               count=10 * SECOND)

timeline_check("late-start-from-2-s", "writes silence from start=2 up to audio that begins at 8 s",
               LATE_BY_8_SECONDS, "start=2", 96000, [silence(288000), real(0, 10 * SECOND)],
               count=10 * SECOND)

timeline_check("start-between-samples", "begins at the first whole sample at or after start",
               LATE_BY_8_SECONDS, "start=0.00001", 1, [silence(383999), real(0, 10 * SECOND)],
               count=10 * SECOND)

timeline_check("lead-of-one-sample", "fills a lead of one sample, which no threshold applies to",
               chain(frames(4800), stamp("PTS+1")), "start=0", 0,
               [silence(1), real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("first-frame-before-start", "keeps audio that begins before start whole",
               LATE_BY_8_SECONDS, "start=10", 384000, [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("first-frame-below-zero", "keeps audio that begins before zero whole",
               chain(frames(4800), stamp("PTS-4800")), "start=0", -4800,
               [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("first-frame-without-timestamp", "begins at start when no frame has a timestamp",
               chain(frames(4800), stamp("NAN")), "start=2", 96000,
               [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("no-frame", "writes nothing when no frame arrives",
               chain(frames(4800), "atrim=start_sample=960000"), "start=0", 0, [],
               count=10 * SECOND)

# --- Limits -------------------------------------------------------------------------------------

timeline_check("max-gap-at-the-limit", "fills a gap as long as max_gap", EIGHT_SECOND_GAP,
               "max_gap=8", 0, EIGHT_SECOND_TIMELINE, count=24 * SECOND)

refusal_check("max-gap-exceeded", "fails on a gap longer than max_gap", EIGHT_SECOND_GAP,
              "max_gap=7.9999", "max_gap", count=24 * SECOND)

refusal_check("max-gap-exceeded-by-lead", "fails on leading silence longer than max_gap",
              LATE_BY_8_SECONDS, "start=0:max_gap=5", "max_gap", count=10 * SECOND)

refusal_check("max-gap-on-2-hours", "fails on a gap of 2 h with max_gap=3600", TWO_HOUR_GAP,
              "max_gap=3600", "max_gap", count=4 * SECOND)

# 10^18 samples, 660,000 years at 48 kHz, are 2.1 x 10^19 microseconds, more than a 64-bit
# timestamp holds. Unless the filter compares a length in samples with one in microseconds in a way
# that cannot overflow, such a gap or lead passes max_gap, or the step does not count as a gap.
refusal_check("max-gap-on-a-lead-of-10-to-the-18", "fails on a lead of 10^18 samples with "
              "max_gap=10", chain(frames(4800), stamp("PTS+1e18")), "start=0:max_gap=10",
              "max_gap", count=4 * SECOND)

refusal_check("max-gap-on-a-gap-of-10-to-the-18", "fails on a gap of 10^18 samples with "
              "max_gap=3600", chain(frames(4800), stamp("PTS+1e18*gte(N,96000)")), "max_gap=3600",
              "max_gap", count=4 * SECOND)

timeline_check("packet-size", "writes silence in frames of packet_size samples", EIGHT_SECOND_GAP,
               "packet_size=1000", 0, EIGHT_SECOND_TIMELINE, count=24 * SECOND, packet_size=1000)

# A gap of 31,000 years. Unless the fill stops when the trim after it has what it needs, FFmpeg
# does not end.
timeline_check("downstream-end", "stops filling when the filter after it needs no more",
               chain(frames(4800), stamp("PTS+48000*1e12*gte(N,96000)")), "", 0,
               [real(0, 96000), silence(1344000)], after="atrim=end_sample=1440000",
               count=4 * SECOND)

@check("end-of-timestamp-range", "stops at the largest timestamp instead of wrapping to the "
       "smallest")
def end_of_timestamp_range(ffmpeg):
    # 3 s of audio stamped from 75,775 samples before the largest timestamp, which is far beyond
    # any real media. Past that value the output cannot stay continuous. It must not go backward.
    counter = COUNTERS["s16"]
    # ffmpeg adds a frame's length to its timestamp. The last filter stamps the frames from
    # zero, so that this sum stays in range.
    shape = chain(frames(4800), stamp("PTS+9.2233720368547e18"), "agapfill", "ashowinfo",
                  stamp("N"))
    command = filter_command(ffmpeg, counter.lavfi(3 * SECOND, RATE), "anull", counter)
    command[command.index("-af") + 1] = shape
    done = execute(command, lambda out: out.read())
    if done.code:
        return ["FFmpeg ended with status %d: %s" % (done.code, last_lines(done.log))]
    stamps = [pts for pts, _, _, _ in frames_of(done.log)]
    if None in stamps:
        return ["%d frames have no timestamp" % stamps.count(None)]
    if not stamps or stamps[0] <= 2 ** 63 - 1 - 3 * SECOND:
        raise Unusable("the audio does not reach the end of the timestamp range")
    problems = []
    backward = [(before, after) for before, after in zip(stamps, stamps[1:]) if after < before]
    if backward:
        problems.append("the frame stamped %d follows the frame stamped %d" % backward[0][::-1])
    width = counter.width
    audio = b"".join(done.result[at:at + width] for at in range(0, len(done.result), width)
                     if any(done.result[at:at + width]))
    if audio != counter.samples(0, 3 * SECOND):
        problems.append("the output does not hold every source sample once and in order")
    return problems


# --- Time bases ---------------------------------------------------------------------------------

timeline_check("millisecond-time-base", "inserts nothing when timestamps are rounded to 1 ms",
               chain(frames(1024), "asettb=1/1000"), "", 0, [real(0, 10 * SECOND)],
               count=10 * SECOND)

timeline_check("millisecond-time-base-gap", "fills a gap exactly when timestamps are in 1 ms",
               chain(EIGHT_SECOND_GAP, "asettb=1/1000"), "", 0, EIGHT_SECOND_TIMELINE,
               count=24 * SECOND)

refusal_check("coarse-time-base", "refuses a threshold that timestamp rounding alone can exceed",
              chain(frames(1024), "asettb=1/100"), "threshold=0.002", "threshold",
              count=10 * SECOND)


# --- Memory -------------------------------------------------------------------------------------

def measured_gap(ffmpeg, shift, runs):
    counter = COUNTERS["s16"]
    shape = chain(frames(4800), stamp("PTS+%d*gte(N,96000)" % shift), "agapfill")
    command = filter_command(ffmpeg, counter.lavfi(4 * SECOND, RATE), shape, counter)
    done = execute(command, lambda out: compare_samples(out, 0, runs, counter, RATE),
                   timeout=5 * TIME_LIMIT, measure=True)
    if done.code:
        return done, ["FFmpeg ended with status %d: %s" % (done.code, last_lines(done.log))]
    return done, check_frames(frames_of(done.log), 0, runs, DEFAULT_PACKET, RATE) + done.result


@check("gap-of-2-hours", "fills a gap of 2 h in the memory a gap of 1 s takes")
def gap_of_two_hours(ffmpeg):
    allowance = 16 << 20
    short, problems = measured_gap(ffmpeg, SECOND, [real(0, 96000), silence(SECOND),
                                                    real(96000, 96000)])
    if problems:
        return ["with a gap of 1 s: " + problem for problem in problems]
    long, problems = measured_gap(ffmpeg, 7200 * SECOND, [real(0, 96000), silence(7200 * SECOND),
                                                          real(96000, 96000)])
    if problems:
        return problems
    megabytes = "at most %.1f MB with a gap of 2 h and %.1f MB with a gap of 1 s" % (
        long.peak() / 2 ** 20, short.peak() / 2 ** 20)
    print("         peak memory of FFmpeg: " + megabytes)
    if long.peak() > short.peak() + allowance:
        return ["peak memory is %s: more than %d MB apart" % (megabytes, allowance >> 20)]
    return []


# --- With an encoder that takes frames of a fixed length ----------------------------------------

def decoded(ffmpeg, path, counter):
    """Decodes a file to (frames, samples) with the timestamps the decoder gives."""
    command = [ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-i", path,
               "-af", "ashowinfo", "-c:a", counter.codec, "-f", counter.raw, "-"]
    done = execute(command, lambda out: out.read())
    if done.code:
        raise Unusable("FFmpeg could not decode %s: %s" % (path, last_lines(done.log)))
    return frames_of(done.log), done.result


def displaced(frame_list, samples, counter):
    """Counts source samples that are not at the time their index names."""
    count, offset, width = 0, 0, counter.width
    for pts, length, _, _ in frame_list:
        got = samples[offset * width:(offset + length) * width]
        offset += length
        if pts < 0 or got == counter.samples(pts, length):
            continue
        expected = counter.samples(pts, length)
        count += sum(1 for i in range(length)
                     if got[i * width:(i + 1) * width] != expected[i * width:(i + 1) * width])
    return count


@check("flac-frames", "keeps every sample in place through a FLAC encode with frames of 4,608")
def flac_frames(ffmpeg):
    counter = COUNTERS["s16"]
    source = chain(counter.lavfi(24 * SECOND, RATE), frames(4800), drop((60, 139), (180, 199)))
    runs = [real(0, 6 * SECOND), silence(8 * SECOND), real(14 * SECOND, 4 * SECOND),
            silence(2 * SECOND), real(20 * SECOND, 4 * SECOND)]
    with tempfile.TemporaryDirectory() as work:
        def encode(name, filters):
            path = os.path.join(work, name)
            done = execute([ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-f", "lavfi",
                            "-i", source, "-af", filters, "-c:a", "flac", "-frame_size", "4608",
                            "-f", "nut", "-y", path])
            if done.code:
                return None, "FFmpeg ended with status %d: %s" % (done.code, last_lines(done.log))
            return path, None

        # Without the fill FFmpeg cuts encoder frames by sample count alone. 288,000 samples
        # precede the first gap, which is 62.5 frames, so 2,304 later samples complete that frame
        # and play 8 s early. 480,000 samples precede the second gap. That leaves 768 in a frame,
        # so 3,840 later samples play 2 s early. A fixture without that defect proves nothing.
        plain, error = encode("plain.nut", "anull")
        if error:
            raise Unusable(error)
        moved = displaced(*decoded(ffmpeg, plain, counter), counter)
        if moved != 2304 + 3840:
            raise Unusable("without the fill %d samples are displaced, expected 6,144" % moved)

        filled, error = encode("filled.nut", "agapfill")
        if error:
            return [error]
        frame_list, samples = decoded(ffmpeg, filled, counter)

    class Bytes:
        def __init__(self, data):
            self.data, self.at = data, 0

        def read(self, size):
            part = self.data[self.at:self.at + size]
            self.at += len(part)
            return part

    return (check_frames(frame_list, 0, runs, 4608, RATE)
            + compare_samples(Bytes(samples), 0, runs, counter, RATE))


# --- With a rate conversion after the fill ------------------------------------------------------

@check("rate-conversion-after-fill", "lets a resampler after the fill keep both sides of a gap "
       "at their own time")
def rate_conversion(ffmpeg):
    # +0.25 of full scale before a gap from 6 s to 14 s and -0.25 after it. A resampler keeps the
    # last input samples in its delay line. Without the fill it writes them when the audio after
    # the gap arrives, stamped just before 14 s.
    out_rate, margin = 44100, 1024
    source = ("aevalsrc=exprs='0.25-0.5*gte(n,%d)':s=%d,atrim=end_sample=%d,"
              "aformat=sample_fmts=s16" % (14 * SECOND, RATE, 24 * SECOND))
    command = [ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-f", "lavfi",
               "-i", source, "-af", chain(EIGHT_SECOND_GAP, "agapfill",
                                          "aresample=osr=%d" % out_rate, "ashowinfo"),
               "-c:a", "pcm_s16le", "-f", "s16le", "-"]
    done = execute(command, lambda out: out.read())
    if done.code:
        return ["FFmpeg ended with status %d: %s" % (done.code, last_lines(done.log))]
    frame_list = frames_of(done.log)
    samples = array.array("h", done.result)
    if sys.byteorder != "little":
        samples.byteswap()
    if not frame_list:
        return ["no frame came out of the resampler"]
    problems = check_frames(frame_list, frame_list[0][0], [real(0, len(samples))], out_rate,
                            out_rate)
    if problems:
        return problems
    first = frame_list[0][0]
    if abs(first) > margin or abs(first + len(samples) - 24 * out_rate) > margin:
        return ["the output covers %d to %d, expected 0 to %d" % (first, first + len(samples),
                                                                 24 * out_rate)]

    def at(begin, end):
        return samples[begin - first:end - first]

    level = 8192
    before = at(5 * out_rate, 6 * out_rate - margin)
    inside = at(6 * out_rate + margin, 14 * out_rate - margin)
    ahead = at(14 * out_rate - margin, 14 * out_rate)
    after = at(14 * out_rate + margin, 15 * out_rate)
    if any(abs(value - level) > 64 for value in before):
        problems.append("the second before the gap is not the audio from before the gap")
    if any(inside):
        problems.append("%d samples inside the gap are not silence"
                        % sum(1 for value in inside if value))
    if any(value > level // 4 for value in ahead):
        problems.append("%d samples just before 14 s hold audio from before the gap"
                        % sum(1 for value in ahead if value > level // 4))
    if any(abs(value + level) > 64 for value in after):
        problems.append("the second after the gap is not the audio from after the gap")
    return problems


# --- Every sample format, three layouts, three rates --------------------------------------------

FORMATS = {"u8": ("pcm_u8", 1), "s16": ("pcm_s16le", 2), "s32": ("pcm_s32le", 4),
           "s64": ("pcm_s64le", 8), "flt": ("pcm_f32le", 4), "dbl": ("pcm_f64le", 8)}
LAYOUTS = {"mono": 1, "5.1": 6, "7.1": 8}


@check("formats", "passes every sample format, layout and rate through and writes each format's "
       "silence")
def formats(ffmpeg):
    problems = []
    for rate in (44100, 48000, 96000):
        frame, total = rate // 100, rate * 3 // 10
        hole = (10 * frame, 20 * frame)
        for layout, channels in LAYOUTS.items():
            tones = "|".join("0.4*sin(2*PI*%d*t)" % (220 * (channel + 1))
                             for channel in range(channels))
            for name in [packed + suffix for packed in FORMATS for suffix in ("", "p")]:
                codec, size = FORMATS[name.rstrip("p")]
                source = ("aevalsrc=exprs='%s':s=%d:c=%s,atrim=end_sample=%d"
                          % (tones, rate, layout, total))

                def output(filters):
                    return execute([ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "info",
                                    "-f", "lavfi", "-i", source, "-af", filters, "-map", "0:a",
                                    "-c:a", codec, "-f", "data", "-"], lambda out: out.read())

                label = "%s %s %d Hz" % (name, layout, rate)
                whole = output("aformat=sample_fmts=%s" % name)
                filled = output(chain("aformat=sample_fmts=%s" % name, frames(frame),
                                      drop((10, 19)), "agapfill", "ashowinfo"))
                if whole.code or len(whole.result) != total * channels * size:
                    raise Unusable("%s: the source did not come out whole" % label)
                if filled.code:
                    problems.append("%s: FFmpeg ended with status %d: %s"
                                    % (label, filled.code, last_lines(filled.log)))
                    continue
                seen = {fmt for _, _, fmt, _ in frames_of(filled.log)}
                if seen != {name}:
                    problems.append("%s: the filter ran in %s" % (label, sorted(seen)))
                width = channels * size
                quiet = (b"\x80" if name.startswith("u8") else b"\x00") * ((hole[1] - hole[0]) * width)
                expected = (whole.result[:hole[0] * width] + quiet + whole.result[hole[1] * width:])
                if filled.result != expected:
                    problems.append("%s: the output is not the source with silence from sample "
                                    "%d to %d" % (label, hole[0], hole[1]))
    return problems


# --- Continuing the output of another agapfill --------------------------------------------------

def resume_command(position, at=0):
    """Sends agapfill the resume command when the first frame at or after `at` seconds passes."""
    return "asendcmd=c='%s agapfill resume %s'" % (at, position)


# The other filter stood at 2 s of output and 8 s of input, counted at 44.1 kHz. The audio begins
# at 8 s, so its first frame is a step of nothing and follows at 2 s.
timeline_check("resume-command", "continues from the position of another filter, given in that "
               "filter's sample rate, and ignores start",
               chain(LATE_BY_8_SECONDS, resume_command("88200/352800/44100")), "start=0",
               2 * SECOND, [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("resume-command-before-gap", "fills the gap between the other filter's last frame "
               "and its own first frame",
               chain(LATE_BY_8_SECONDS, resume_command("96000/96000/48000")), "", 2 * SECOND,
               [silence(6 * SECOND), real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("resume-command-after-first-frame", "ignores the resume command once it has taken "
               "a frame", chain(LATE_BY_8_SECONDS, resume_command("0/0/48000", at=9)), "",
               8 * SECOND, [real(0, 10 * SECOND)], count=10 * SECOND)

timeline_check("resume-command-incomplete", "starts afresh when the resume command names no "
               "whole position", chain(LATE_BY_8_SECONDS, resume_command("96000/96000")), "start=0", 0,
               [silence(8 * SECOND), real(0, 10 * SECOND)], count=10 * SECOND)


# --- When ffmpeg rebuilds the filter graph ------------------------------------------------------

def timeline_by_the_rules(frame_list, start=None, threshold=RATE * 2 // 1000):
    """The first timestamp and the runs that rules 1 to 4 give for input frames."""
    runs, first, position, end, taken = [], None, None, None, 0
    for pts, length, _, _ in frame_list:
        quiet = 0
        if end is None:
            first = position = pts if start is None else min(start, pts)
            quiet = pts - position
        elif pts - end > threshold:
            quiet = max(pts - position, 0)
        if quiet:
            runs.append(silence(quiet))
        runs.append(("input", taken, length))
        position += quiet + length
        end = pts + length
        taken += length
    return first, runs


def changing_layout(ffmpeg, work, hole):
    """Writes 5 s of 5.1 AC-3 and then 5 s of stereo AC-3, stamped `hole` seconds after the 5.1."""
    # The encoder writes 5 s as 157 frames of 1,536 samples. Both parts are stamped from 1 s or
    # later, because the muxer would move a part that begins below zero.
    stream, end = b"", 157 * 1536 / RATE
    for layout, channels, offset in (("5.1", 6, 1), ("stereo", 2, 1 + end + hole)):
        tones = "|".join("0.3*sin(2*PI*%d*t)" % (220 * (channel + 1)) for channel in range(channels))
        done = execute([ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                        "aevalsrc=exprs='%s':s=%d:c=%s:d=5" % (tones, RATE, layout),
                        "-c:a", "ac3", "-b:a", "384k", "-output_ts_offset", "%.6f" % offset,
                        "-f", "mpegts", "-"], lambda out: out.read())
        if done.code:
            raise Unusable("FFmpeg could not encode AC-3: %s" % last_lines(done.log))
        stream += done.result
    path = os.path.join(work, "changing-layout.ts")
    with open(path, "wb") as out:
        out.write(stream)
    return path


def rebuild_problems(ffmpeg, *, hole, ahead="anull", start=None, seek=None):
    """Fills audio that changes layout and compares the output with the rules' timeline.

    `ahead` is the filter chain ahead of the fill. The reference is the same decode without the
    fill: its frames are the input, and its samples are the real audio. ffmpeg builds a new filter
    graph at the change of layout.
    """
    fill = "agapfill" if start is None else "agapfill=start=%d" % start
    with tempfile.TemporaryDirectory() as work:
        source = changing_layout(ffmpeg, work, hole)

        def decode(filters):
            position = ["-ss", str(seek)] if seek is not None else []
            return execute([ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "info"] + position
                           + ["-i", source, "-af", chain(filters, "ashowinfo"), "-ac", "2",
                              "-c:a", "pcm_s16le", "-f", "s16le", "-"], lambda out: out.read())

        plain, filled = decode(ahead), decode(chain(ahead, fill))
    if plain.code or "Reconfiguring filter graph" not in plain.log:
        raise Unusable("ffmpeg did not rebuild the filter graph at the change of layout")
    if filled.code:
        return ["FFmpeg ended with status %d: %s" % (filled.code, last_lines(filled.log))]
    first, runs = timeline_by_the_rules(frames_of(plain.log),
                                        None if start is None else start * RATE)
    problems = check_frames(frames_of(filled.log), first, runs, DEFAULT_PACKET, RATE)
    expected = b"".join(bytes(4 * count) if kind == "silence"
                        else plain.result[4 * at:4 * (at + count)] for kind, at, count in runs)
    quiet = sum(count for kind, _, count in runs if kind == "silence")
    if len(filled.result) != len(expected):
        problems.append("the output holds %d samples, expected %d of silence and %d of source"
                        % (len(filled.result) // 4, quiet, len(plain.result) // 4))
    elif filled.result != expected:
        at = next(at for at in range(0, len(expected), 4)
                  if filled.result[at:at + 4] != expected[at:at + 4])
        problems.append("the sample at %d (%.6f s) is not the one the timeline has there"
                        % (first + at // 4, (first + at // 4) / RATE))
    return problems


@check("graph-rebuild", "writes the leading silence once when ffmpeg rebuilds the filter graph")
def graph_rebuild(ffmpeg):
    # An AC-3 stream that changes from 5.1 to stereo after 5 s, stamped 2 s late. Unless the new
    # graph's filter continues the old one's output, it writes silence from zero a second time.
    return rebuild_problems(ffmpeg, hole=0, ahead="asetpts=PTS+%d" % (2 * SECOND), start=0)


@check("overlap-at-graph-rebuild", "keeps the output continuous when timestamps overlap where "
       "ffmpeg rebuilds the filter graph")
def overlap_at_graph_rebuild(ffmpeg):
    # The stereo part is stamped 24 ms before the 5.1 part ends. Unless the new graph's filter
    # continues the old one's output, its first frame steps back by those 24 ms.
    return rebuild_problems(ffmpeg, hole=-0.024)


@check("gap-at-graph-rebuild", "fills a gap that falls where ffmpeg rebuilds the filter graph")
def gap_at_graph_rebuild(ffmpeg):
    # The stereo part is stamped 3 s after the 5.1 part ends. Only the old graph's filter knows
    # where the audio stopped. Unless the new one learns it, the 3 s stay a hole.
    return rebuild_problems(ffmpeg, hole=3)


@check("seek-past-graph-rebuild", "writes the leading silence when ffmpeg rebuilds the filter "
       "graph before the first frame arrives")
def seek_past_graph_rebuild(ffmpeg):
    # A seek to 6 s, inside a hole from 5.024 s to 8.024 s. ffmpeg decodes from before the change
    # of layout and discards those frames ahead of the filter, so the first graph's filter never
    # sees a frame. The audio begins 2.024 s after the seek point, and the new graph's filter owes
    # the silence from start, 1 s after the seek point.
    return rebuild_problems(ffmpeg, hole=3, start=1, seek=6)


CHANGING_SIZE = (("64x36", 10), ("128x72", 30))
STEADY_SIZE = (("64x36", 40),)


def beside_video(ffmpeg, shape, option, first, runs, after="anull", audio_frames=None,
                 sizes=CHANGING_SIZE):
    """Fills the counter in one filter graph with a video made of `sizes`, (frame size, seconds).

    The audio file carries the timestamps `shape` gives. By default the frame size changes at
    10 s. ffmpeg reads the first frame after the hole, or the late first frame, while the video
    is still short of 10 s, so the filter is writing silence when the change of size makes ffmpeg
    free the graph and build a new one. `after` is the filter behind the fill. The audio output
    takes at most `audio_frames` frames.
    """
    limit = [] if audio_frames is None else ["-frames:a", str(audio_frames)]
    counter = COUNTERS["s16"]
    quiet = ["-nostdin", "-hide_banner", "-v", "error"]
    with tempfile.TemporaryDirectory() as work:
        video, audio = os.path.join(work, "video.mjpeg"), os.path.join(work, "audio.nut")
        with open(video, "wb") as out:
            for size, seconds in sizes:
                part = execute([ffmpeg] + quiet + ["-f", "lavfi", "-i",
                                                   "testsrc2=s=%s:r=24:d=%d" % (size, seconds),
                                                   "-c:v", "mjpeg", "-q:v", "20", "-f", "mjpeg", "-"],
                               lambda stream: stream.read())
                if part.code:
                    raise Unusable("FFmpeg could not encode the video: %s" % last_lines(part.log))
                out.write(part.result)
        made = execute([ffmpeg] + quiet + ["-f", "lavfi", "-i", counter.lavfi(40 * SECOND, RATE),
                                           "-af", shape, "-c:a", counter.codec, "-f", "nut", "-y",
                                           audio])
        if made.code:
            raise Unusable("FFmpeg could not write the audio: %s" % last_lines(made.log))
        graph = "[0:v]scale=64:36[v];[1:a]agapfill%s,%s,ashowinfo[a]" % (option, after)
        done = execute([ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-copyts",
                        "-f", "mjpeg", "-framerate", "24", "-i", video, "-i", audio,
                        "-filter_complex", graph, "-map", "[v]", "-f", "null", "-",
                        "-map", "[a]"] + limit + ["-c:a", counter.codec, "-f", counter.raw, "-"],
                       lambda out: compare_samples(out, first, runs, counter, RATE))
    if done.code:
        return ["FFmpeg ended with status %d: %s" % (done.code, last_lines(done.log))]
    rebuilt = "Reconfiguring filter graph because video parameters changed" in done.log
    if len(sizes) > 1 and not rebuilt:
        raise Unusable("ffmpeg did not rebuild the filter graph at the change of frame size")
    return check_frames(frames_of(done.log), first, runs, DEFAULT_PACKET, RATE) + done.result


@check("rebuild-during-gap", "keeps every sample and the whole gap when ffmpeg rebuilds the filter "
       "graph while the filter writes the gap")
def rebuild_during_gap(ffmpeg):
    return beside_video(ffmpeg, chain(frames(4800), drop((20, 319))), "", 0,
                        [real(0, 2 * SECOND), silence(30 * SECOND), real(32 * SECOND, 8 * SECOND)])


# The trim ends the audio at 12 s, inside the gap. At the rebuild the filter still holds the frame
# from after the gap, and no output asks for it any more. Unless ffmpeg stops asking, it never ends.
@check("rebuild-during-gap-behind-end", "lets ffmpeg rebuild the filter graph when the filter "
       "behind it needs no more of the gap")
def rebuild_during_gap_behind_end(ffmpeg):
    return beside_video(ffmpeg, chain(frames(4800), drop((20, 319))), "", 0,
                        [real(0, 2 * SECOND), silence(10 * SECOND)], after="atrim=end=12")


# The audio output takes 20 frames, the 2 s before a gap of 31,000 years, and is then finished. One
# last frame follows the gap. No filter ends. Unless ffmpeg stops asking an output that is finished
# for frames, it writes the whole gap before it builds the new graph, and it never ends.
@check("rebuild-during-gap-behind-finished-output", "lets ffmpeg rebuild the filter graph when the "
       "output behind the filter takes no more frames")
def rebuild_during_gap_behind_finished_output(ffmpeg):
    shape = chain(frames(4800), drop((21, 399)), stamp("PTS+48000*1e12*gte(N,96000)"))
    return beside_video(ffmpeg, shape, "", 0, [real(0, 2 * SECOND)], audio_frames=20)


@check("rebuild-during-lead", "keeps every sample and the whole lead when ffmpeg rebuilds the "
       "filter graph while the filter writes the leading silence")
def rebuild_during_lead(ffmpeg):
    return beside_video(ffmpeg, chain(frames(4800), stamp("PTS+%d" % (20 * SECOND))),
                        "=start=0", 0, [silence(20 * SECOND), real(0, 40 * SECOND)])


CHECKS_BY_NAME = {name: (title, function) for name, title, function in CHECKS}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ffmpeg", help="the ffmpeg binary to check")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="run only these checks")
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    arguments = parser.parse_args()
    if arguments.list:
        for name, title, _ in CHECKS:
            print("%-32s %s" % (name, title))
        return 0
    if not arguments.ffmpeg:
        parser.error("--ffmpeg is required")
    unknown = [name for name in arguments.only or [] if name not in CHECKS_BY_NAME]
    if unknown:
        parser.error("unknown checks: %s" % ", ".join(unknown))

    failed, unusable = 0, 0
    selected = [entry for entry in CHECKS if not arguments.only or entry[0] in arguments.only]
    for name, title, function in selected:
        try:
            problems = function(arguments.ffmpeg)
        except Unusable as reason:
            unusable += 1
            print("UNUSABLE %s: %s\n     %s" % (name, title, reason), flush=True)
            continue
        if problems:
            failed += 1
            print("FAILED   %s: agapfill %s\n     %s" % (name, title, "\n     ".join(problems)),
                  flush=True)
        else:
            print("ok       %s: agapfill %s" % (name, title), flush=True)
    print("%d of %d checks failed, %d unusable" % (failed, len(selected), unusable))
    if unusable:
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
