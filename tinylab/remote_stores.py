"""
datacore `DatasetStore` wrappers that talk to the bucket. datacore already accepts any store, so
none of this needs a datacore change.

- `UploadingDatasetStore` (prepare): each written shard is queued for upload right away; the
  manifest goes last (datacore itself writes it last locally, and the Uploader skips a marker whose
  entity had a failed upload) -- so the bucket never shows a dataset as complete before all its
  shards are there.
- `PrefetchingDatasetStore` (train): `open_volume` blocks until the shard is local and moves a
  lookahead window forward, so training starts on the first shards while later ones still download.
"""
import os

from datacore import FileSystemDatasetStore

from tinylab import remote as remote_mod


class UploadingDatasetStore(FileSystemDatasetStore):
    def __init__(self, dataset_dir: str, *, name: str, uploader, producer: dict, inputs: dict | None = None,
                 motivation: str = "", force: bool = False):
        super().__init__(dataset_dir)
        self.entity = f"{remote_mod.PREPARED}/{name}"
        self._uploader, self._producer, self._inputs = uploader, producer, inputs or {}
        self._motivation, self._force = motivation, force
        self._listing: dict | None = None
        self._existing_readme = None
        self._gated = False

    def _state(self, remote):
        """First task of the run: producer gate + one remote listing, shared by every shard task."""
        if not self._gated:
            self._existing_readme = remote_mod.gate_producer(remote, self.entity, self._producer, force=self._force)
            self._listing = remote.list(self.entity)
            self._gated = True
        return self._listing

    def write_volume(self, filename: str, array) -> None:
        super().write_volume(filename, array)
        local = os.path.join(self.dataset_dir, filename)

        def task(remote):
            listing = self._state(remote)
            # A shard is never "complete" on its own: only the manifest makes the group complete.
            remote_mod.push_group(remote, listing, [(local, f"{self.entity}/{filename}")], None, force=self._force)

        self._uploader.submit(self.entity, task, label=f"{self.entity}/{filename}")

    def write_manifest(self, manifest: dict) -> None:
        super().write_manifest(manifest)
        local = self._manifest_path()

        def task(remote):
            listing = self._state(remote)
            remote_mod.push_group(remote, listing, [], (local, f"{self.entity}/manifest.json"), force=self._force)
            summary = {split: {"num_sequences": s.get("num_sequences"), "num_tokens": s.get("num_tokens")}
                       for split, s in (manifest.get("splits") or {}).items()}
            remote_mod.write_readme(remote, self.entity, self._existing_readme, kind="dataset", producer=self._producer,
                                    inputs=self._inputs, motivation=self._motivation, metrics=summary,
                                    extra={"sequence_len": manifest.get("sequence_len")},
                                    history="pushed (complete)")

        self._uploader.submit(self.entity, task, marker=True, label=f"{self.entity}/manifest.json")


class PrefetchingDatasetStore(FileSystemDatasetStore):
    """Reads a prepared dataset, fetching whatever isn't local from `prepared/<name>/` in the bucket.
    Only reads: the local folder is filled as a side effect, atomically, file by file."""

    def __init__(self, dataset_dir: str, *, name: str, remote, lookahead: int = 2):
        super().__init__(dataset_dir)
        self.entity = f"{remote_mod.PREPARED}/{name}"
        self._remote, self._lookahead = remote, lookahead
        self._prefetcher = None

    def read_manifest(self):
        path = self._manifest_path()
        if not os.path.exists(path):
            if f"{self.entity}/manifest.json" not in self._remote.list(self.entity):
                return None  # not in the bucket either (or incomplete: no marker) -> "not prepared"
            self._remote.download([(f"{self.entity}/manifest.json", path)])
        manifest = super().read_manifest()
        if manifest is not None and self._prefetcher is None:
            self._start(manifest)
        return manifest

    def _start(self, manifest: dict) -> None:
        train, val, extra = [], [], []
        for split, data in manifest["splits"].items():
            names = []
            for v in data["volumes"]:
                names.append(v["file"])
                if v.get("mask_file"):
                    names.append(v["mask_file"])
            (val if split == "val" else train).extend(names)
        if manifest.get("token_bytes_file"):
            extra.append(manifest["token_bytes_file"])
        per_volume = 2 if any(v.get("mask_file") for d in manifest["splits"].values() for v in d["volumes"]) else 1
        self._prefetcher = remote_mod.Prefetcher(self._remote, self.entity, self.dataset_dir, extra + val + train,
                                                 lookahead=self._lookahead * per_volume)
        # Small and needed before the first step / first eval: fetch them up front.
        self._prefetcher.fetch_all(extra + val)

    def open_volume(self, filename: str, mmap: bool = True):
        if self._prefetcher is not None:
            self._prefetcher.ensure(filename)
        return super().open_volume(filename, mmap=mmap)

    def close(self):
        if self._prefetcher is not None:
            self._prefetcher.close()


