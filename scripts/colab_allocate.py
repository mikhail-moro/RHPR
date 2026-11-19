"""Run the pinned Colab CLI with structured allocation failure diagnostics."""

import os
import sys
import time
import traceback
from contextlib import ExitStack
from unittest.mock import patch

from colab_cli.client import Client, ColabRequestError
from colab_cli.cli import app
from colab_cli.utils import get_status_code
from requests.exceptions import ReadTimeout


RETRY_DELAYS_SECONDS = (60, 120, 240, 480)


def retry_assignment(assign):
    def assign_with_backoff(client, notebook_hash, **options):
        for delay in (*RETRY_DELAYS_SECONDS, 0):
            try:
                # The same notebook ID lets the pinned CLI look up an allocation
                # before posting again, even after an uncertain HTTP response.
                return assign(client, notebook_hash, **options)
            except (ColabRequestError, ReadTimeout) as error:
                status = get_status_code(error)
                if not delay or (status not in {502, 503, 504} and not isinstance(error, ReadTimeout)):
                    raise
                reason = f"HTTP {status}" if status is not None else "read timeout"
                print(f"Colab allocation returned {reason}; retrying in {delay}s", file=sys.stderr, flush=True)
                time.sleep(delay)

    return assign_with_backoff


def main():
    args = sys.argv[1:]
    try:
        with ExitStack() as stack:
            if args[:1] == ["--retry-transient"]:
                args = args[1:]
                stack.enter_context(patch.dict(os.environ, {"COLAB_CLI_ASSIGN_RETRIES": "0",
                                                           "COLAB_CLI_REQUEST_TIMEOUT": "45"}))
                stack.enter_context(patch.object(Client, "assign", retry_assignment(Client.assign)))
            return app(args=args, standalone_mode=False)
    except Exception as error:
        status = get_status_code(error)
        if isinstance(status, int) and 400 <= status < 600:
            print(f"Colab backend HTTP {status}", file=sys.stderr)
        print(f"Colab allocation exception: {type(error).__name__}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
