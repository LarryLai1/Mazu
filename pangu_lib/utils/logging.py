import sys
from datetime import datetime
from typing import Any


def log_message(message: Any) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", file=sys.stderr)
