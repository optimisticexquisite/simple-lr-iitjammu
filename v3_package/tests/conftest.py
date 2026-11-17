# Lets the tests "import linreg" no matter where pytest is started from.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
