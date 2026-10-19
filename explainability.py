"""
Root backward-compatibility proxy for src.explainability.
"""
from src.explainability import *
import src.explainability as explainability

if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).parent / "src" / "explainability.py"), run_name="__main__")
