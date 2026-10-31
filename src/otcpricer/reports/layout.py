"""Shared formatting for the plain-text validation reports.

One place for the house style, so the surface report and the local-volatility
report cannot drift apart in width or wrapping.

Plain text rather than HTML or a notebook, deliberately: a validation report
has to be readable in a terminal, diffable in git, and quotable in a commit
message. Numbers are rendered at the precision they are actually known to --
volatility to a hundredth of a point, not to six decimals that imply a
confidence the inputs do not support.
"""

from __future__ import annotations

import textwrap

__all__ = ["WIDTH", "rule", "wrap"]

#: Report width. Fits a terminal without wrapping, and leaves room for the
#: two-space indent the body uses.
WIDTH = 78


def rule(char: str = "-") -> str:
    """A horizontal rule the full width of the report."""
    return char * WIDTH


def wrap(text: str, indent: str = "  ") -> list[str]:
    """Reflow prose to the report width.

    Returns a list of lines so callers can splice it into a section without
    worrying about trailing newlines. Never returns an empty list, so a blank
    input still occupies its line rather than collapsing a section.
    """
    return textwrap.wrap(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent) or [
        indent.rstrip()
    ]
