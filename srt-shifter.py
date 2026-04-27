import re
import sys
from pathlib import Path
from datetime import timedelta


def parse_time(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return timedelta(
        hours=int(h),
        minutes=int(m),
        seconds=int(s),
        milliseconds=int(ms)
    )


def format_time(td):
    total_ms = max(0, int(td.total_seconds() * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def shift_srt(input_file, output_file, offset_seconds):
    text = Path(input_file).read_text(encoding="utf-8")

    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
    )

    offset = timedelta(seconds=offset_seconds)

    def replace(match):
        start = parse_time(match.group(1)) + offset
        end = parse_time(match.group(2)) + offset
        return f"{format_time(start)} --> {format_time(end)}"

    new_text = pattern.sub(replace, text)
    Path(output_file).write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python shift_srt.py input.srt output.srt offset_seconds")
        print("Example: python shift_srt.py old.srt synced.srt 2.5")
        print("Example: python shift_srt.py old.srt synced.srt -1.2")
        sys.exit(1)

    shift_srt(sys.argv[1], sys.argv[2], float(sys.argv[3]))