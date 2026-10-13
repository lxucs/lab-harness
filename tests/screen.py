"""Replays terminal output into the screen a user would see.

The renderer redraws a live region in place and erases it when a run ends, so
the raw bytes hold every intermediate frame. Counting a substring in them
cannot tell text left in scrollback from a frame that was drawn and then wiped;
replaying the cursor movement can.

Covers what rich and prompt_toolkit emit — cursor movement, erase in line and
below, carriage return — and ignores styling and cursor visibility. Every
character is taken to be one cell wide, which is enough to compare text.
"""

import re

_CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
#: Operating-system commands, e.g. hyperlinks, ended by ST or BEL.
_OSC = re.compile(r"\x1b\].*?(?:\x1b\\|\x07)", re.DOTALL)


def screen(output: str) -> list[str]:
    """The lines left on screen and in scrollback, trailing spaces trimmed."""
    lines: list[list[str]] = [[]]
    row = col = 0
    i = 0
    while i < len(output):
        if output.startswith("\x1b]", i):
            match = _OSC.match(output, i)
            i = match.end() if match else len(output)
            continue
        if output[i] == "\x1b":
            match = _CSI.match(output, i)
            if match is None:
                i += 1
                continue
            params, final = match.groups()
            count = int(params) if params.isdigit() else 1
            if final == "A":
                row = max(0, row - count)
            elif final == "B":
                row += count
            elif final == "C":
                col += count
            elif final == "D":
                col = max(0, col - count)
            elif final == "G":
                col = count - 1
            elif final == "K":
                while len(lines) <= row:
                    lines.append([])
                line = lines[row]
                if params in ("", "0"):
                    del line[col:]
                elif params == "1":
                    line[: col + 1] = [" "] * min(col + 1, len(line))
                elif params == "2":
                    line.clear()
            elif final == "J" and params in ("", "0") and row < len(lines):
                # Erase below, which is how prompt_toolkit clears its prompt.
                del lines[row][col:]
                del lines[row + 1 :]
            i = match.end()
            continue
        char = output[i]
        i += 1
        if char == "\n":
            row += 1
            col = 0
            continue
        if char == "\r":
            col = 0
            continue
        while len(lines) <= row:
            lines.append([])
        line = lines[row]
        line.extend(" " * (col - len(line)))
        if col < len(line):
            line[col] = char
        else:
            line.append(char)
        col += 1
    rendered = ["".join(line).rstrip() for line in lines]
    while rendered and not rendered[-1]:
        rendered.pop()
    return rendered


def visible(output: str) -> str:
    return "\n".join(screen(output))
