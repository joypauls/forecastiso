import os
import sys
import contextlib


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = fnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
