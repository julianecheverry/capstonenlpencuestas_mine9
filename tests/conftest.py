# conftest.py
"""
Shared pytest configuration for the whole test suite.

The project now has a two-level source layout:

    src/
    |-- settings.py          (shared configuration, reads Excel files)
    |-- datacleaning/
        |-- limpieza.py      (Cleaner class)
        |-- synonyms.py      (SynonymReplacer class)

`limpieza.py` and `synonyms.py` both do `from settings import ...`, which
requires `src/` to be importable, while the tests themselves import
`from limpieza import Cleaner` / `from synonyms import SynonymReplacer`,
which requires `src/datacleaning/` to be importable. Both paths are added
here once, to avoid repeating the sys.path manipulation in every test
file.

NOTE: importing `settings` (directly or indirectly through limpieza /
synonyms) reads two Excel files from `data/utilities/` at import time
(`accents.xlsx` and `stopwords.xlsx`). These files are versioned in the
repository. If they are missing, every
test in the suite will fail at collection time with an import error --
that is by design of settings.py (a testability finding documented in
HALLAZGOS_Y_TESTING.md).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "datacleaning"))
