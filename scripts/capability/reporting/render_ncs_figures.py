#!/usr/bin/env python3
"""Render the supplementary figures through the shared manuscript renderer.

The previous duplicate five-main-figure layout is retired. All typography and
source-row handling now live in render_figures.py; this entry point is retained.
"""
import sys
from render_figures import main

if __name__ == '__main__':
    sys.argv.append('--supplement-only')
    main()
