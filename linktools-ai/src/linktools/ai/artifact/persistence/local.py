"""Content-addressed local ArtifactStore."""

import asyncio
import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path

from ..models import ArtifactProvenance, ArtifactRecord, ArtifactRef
from ...execution.codec import decode, encode


class LocalArtifactStore:
    def __init__(self, root: str | Path = ".linktools") -> None:
        self.root = Path(root)

    async def put(self, *, record: ArtifactRecord, content) -> ArtifactRecord:
        digest, size, temporary = await asyncio.to_thread(self._stage, content)
        if digest != record.ref.sha256 or size != record.ref.size:
            await asyncio.to_thread(temporary.unlink, missing_ok=True)
            raise ValueError("artifact digest or size mismatch")
        blob = self.root / "artifacts" / "blobs" / digest
        metadata = self.root / "artifacts" / "records" / f"{record.ref.id}.json"
        await asyncio.to_thread(self._put_sync, blob, metadata, record, temporary)
        return record

    @staticmethod
    def _stage(content) -> tuple[str, int, Path]:
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(prefix="linktools-artifact-", delete=False) as target:
            temporary = Path(target.name)
            async def consume() -> None:
                nonlocal size
                async for chunk in content:
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
            asyncio.run(consume())
        return digest.hexdigest(), size, temporary

    @staticmethod
    def _put_sync(blob: Path, metadata: Path, record: ArtifactRecord, temporary: Path) -> None:
        blob.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            temporary.replace(blob)
        else:
            temporary.unlink(missing_ok=True)
        metadata.write_text(encode(asdict(record)), encoding="utf-8")

    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        path = self.root / "artifacts" / "records" / f"{artifact_id}.json"
        if not path.exists():
            return None
        raw = await asyncio.to_thread(lambda: decode(path.read_text(encoding="utf-8")))
        raw["ref"] = ArtifactRef(**raw["ref"])
        raw["provenance"]["parent_artifact_ids"] = tuple(raw["provenance"].get("parent_artifact_ids", ()))
        raw["provenance"] = ArtifactProvenance(**raw["provenance"])
        return ArtifactRecord(**raw)

    async def open(self, artifact_id: str):
        record = await self.get_record(artifact_id)
        if record is None:
            return
        path = self.root / "artifacts" / "blobs" / record.ref.sha256
        with path.open("rb") as source:
            while chunk := await asyncio.to_thread(source.read, 1024 * 1024):
                yield chunk

    async def delete(self, artifact_id: str) -> None:
        record_path = self.root / "artifacts" / "records" / f"{artifact_id}.json"
        record = await self.get_record(artifact_id)
        if record is None:
            return
        await asyncio.to_thread(record_path.unlink, missing_ok=True)
