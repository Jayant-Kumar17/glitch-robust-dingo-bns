#!/usr/bin/env python3
"""Task 5 journal figures — alias of paper/make_figures.py.

paper/main.tex (Appendix C / Data availability) cites both
``paper/make_figures.py`` and ``paper/make_figures_t5.py``. They generate the
same suite (filenames fig1, fig4–fig9, S1–S4). Paper Figures 4 and 5
(filenames fig2, fig3) come from examples/paper_figures_2_3.py.
"""
from pathlib import Path
import runpy
import sys

_target = Path(__file__).with_name("make_figures.py")
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
