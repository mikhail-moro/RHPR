"""Run the pinned Colab CLI with structured allocation failure diagnostics."""

import sys
import traceback

from colab_cli.cli import app
from colab_cli.utils import get_status_code


def main():
    try:
        return app(standalone_mode=False)
    except Exception as error:
        status = get_status_code(error)
        if isinstance(status, int) and 400 <= status < 600:
            print(f"Colab backend HTTP {status}", file=sys.stderr)
        print(f"Colab allocation exception: {type(error).__name__}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
