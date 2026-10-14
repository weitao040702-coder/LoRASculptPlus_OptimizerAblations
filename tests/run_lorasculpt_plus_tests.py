"""Small dependency-free test runner for this environment."""

import importlib.util
import traceback
from pathlib import Path


path = Path(__file__).with_name("test_lorasculpt_plus.py")
spec = importlib.util.spec_from_file_location("test_lorasculpt_plus", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

tests = sorted(name for name in dir(module) if name.startswith("test_"))
for name in tests:
    try:
        getattr(module, name)()
    except Exception:
        traceback.print_exc()
        raise SystemExit(f"FAILED: {name}")
    print(f"PASS: {name}")
print(f"{len(tests)} tests passed")
