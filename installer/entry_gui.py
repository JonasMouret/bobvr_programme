"""Entry point for the packaged window.

PyInstaller needs a script, not a console-script name, and freezing wants
multiprocessing's guard in place before anything else runs.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from bobvr.ui.app import main

    sys.exit(main())
