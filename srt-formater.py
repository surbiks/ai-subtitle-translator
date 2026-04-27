import re
import sys
from pathlib import Path


def flatten_text_to_srt(input_text: str) -> str:
    """
    Convert flattened SRT-like text into properly formatted SRT.
    Example input:
    1 00:00:04,738 --> 00:00:07,573 [dramatic music] 2 00:00:07,674 ...
    """

    pattern = re.compile(
        r'(\d+)\s+'
        r'(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+'
        r'(\d{2}:\d{2}:\d{2},\d{3})\s+'
    )

    matches = list(pattern.finditer(input_text))

    if not matches:
        raise ValueError("No valid SRT timestamp blocks found.")

    srt_blocks = []

    for i, match in enumerate(matches):
        subtitle_number = match.group(1)
        start_time = match.group(2)
        end_time = match.group(3)

        text_start = match.end()

        if i + 1 < len(matches):
            text_end = matches[i + 1].start()
        else:
            text_end = len(input_text)

        subtitle_text = "\n".join(
            line.strip()
            for line in input_text[text_start:text_end].strip().splitlines()
        )

        block = f"{subtitle_number}\n{start_time} --> {end_time}\n{subtitle_text}"
        srt_blocks.append(block)

    return "\n\n".join(srt_blocks) + "\n"


def main():
    if len(sys.argv) != 3:
        print("Usage: python format_srt.py input.txt output.srt")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    input_text = input_path.read_text(encoding="utf-8")
    formatted_srt = flatten_text_to_srt(input_text)

    output_path.write_text(formatted_srt, encoding="utf-8")

    print(f"Formatted SRT saved to: {output_path}")


if __name__ == "__main__":
    main()
