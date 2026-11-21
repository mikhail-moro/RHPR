"""Read the bounded generator checkpoint archive used for GPU continuations."""
import shutil
import tarfile


CHECKPOINT_FILES = {
    "generator-last.pt": 512 * 1024**2,
    "generator-best.pt": 256 * 1024**2,
    "training-curves.json": 4 * 1024**2,
}


def unpack_checkpoint(source, destination):
    with tarfile.open(source, "r:") as archive:
        members = archive.getmembers()
        if (len(members) != len(CHECKPOINT_FILES)
                or {member.name for member in members} != CHECKPOINT_FILES.keys()
                or any(not member.isfile() or not 0 < member.size <= CHECKPOINT_FILES[member.name]
                       for member in members)):
            raise ValueError("expected the last checkpoint, best checkpoint and curves in checkpoint.tar")
        destination.mkdir()
        for member in members:
            with archive.extractfile(member) as stream, (destination / member.name).open("xb") as target:
                shutil.copyfileobj(stream, target)
