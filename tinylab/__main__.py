"""
The entire tinylab CLI:

    python -m tinylab <job.json> [--only NAME] [--dry-run]   # run the pipeline
    python -m tinylab chat <job.json>                        # interactive chat

Every other knob is a JSON key in the job file -- see README.md.
"""
import sys

from tinylab import job as job_module


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2

    if argv[0] == "chat":
        if len(argv) != 2:
            print("usage: python -m tinylab chat <job.json>", file=sys.stderr)
            return 2
        from tinylab import chat
        try:
            chat.main(argv[1])
        except job_module.JobError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    job_path = argv[0]
    only = None
    dry_run = False
    rest = argv[1:]
    i = 0
    while i < len(rest):
        if rest[i] == "--only":
            i += 1
            if i >= len(rest):
                print("error: --only requires a step name\n\n" + __doc__, file=sys.stderr)
                return 2
            only = rest[i]
        elif rest[i] == "--dry-run":
            dry_run = True
        else:
            print(f"unrecognized argument: {rest[i]!r}\n\n{__doc__}", file=sys.stderr)
            return 2
        i += 1

    try:
        job_module.run_file(job_path, only=only, dry_run=dry_run)
    except job_module.JobError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
