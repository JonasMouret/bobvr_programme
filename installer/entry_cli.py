"""Entry point for the packaged command line tool."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from bobvr.__main__ import main

    sys.exit(main())
