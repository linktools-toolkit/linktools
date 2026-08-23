"""Validated inline/object payload descriptors for versioned records."""

import base64
import binascii
import hashlib
from dataclasses import dataclass

from ..core import JsonValue, canonical_json_bytes
from ._object import ObjectRef

_MAX_INLINE_LIMIT = 256 * 1024
_DIGEST_SIZE = 64


@dataclass(frozen=True, slots=True)
class PayloadPolicy:
    inline_limit_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if (
            not isinstance(self.inline_limit_bytes, int)
            or isinstance(self.inline_limit_bytes, bool)
            or not 0 <= self.inline_limit_bytes <= _MAX_INLINE_LIMIT
        ):
            raise ValueError("inline payload limit must be between 0 and 256 KiB")


@dataclass(frozen=True, slots=True)
class StoredPayload:
    kind: str
    encoding: str | None
    digest: str
    size: int
    value: JsonValue = None
    ref: "ObjectRef | None" = None

    def __post_init__(self) -> None:
        if self.kind not in {"inline", "object"}:
            raise ValueError("payload kind is invalid")
        if not isinstance(self.digest, str) or len(self.digest) != _DIGEST_SIZE or self.digest.lower() != self.digest:
            raise ValueError("payload digest is invalid")
        try:
            int(self.digest, 16)
        except ValueError as error:
            raise ValueError("payload digest is invalid") from error
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("payload size is invalid")
        if self.kind == "inline":
            if self.encoding not in {"json", "utf-8", "base64"} or self.ref is not None:
                raise ValueError("inline payload descriptor is invalid")
            actual_digest, actual_size = _inline_digest_size(self.encoding, self.value)
            if actual_digest != self.digest or actual_size != self.size:
                raise ValueError("inline payload digest or size does not match value")
        elif self.encoding is not None or self.value is not None or self.ref is None:
            raise ValueError("object payload descriptor is invalid")
        else:
            if not isinstance(self.ref, ObjectRef):
                raise TypeError("object payload reference is invalid")
            if self.ref.digest != self.digest or self.ref.size != self.size:
                raise ValueError("object payload descriptor does not match reference")

    @classmethod
    def inline_json(cls, value: JsonValue) -> "StoredPayload":
        digest, size = _inline_digest_size("json", value)
        return cls("inline", "json", digest, size, value)

    @classmethod
    def inline_text(cls, value: str) -> "StoredPayload":
        digest, size = _inline_digest_size("utf-8", value)
        return cls("inline", "utf-8", digest, size, value)

    @classmethod
    def inline_bytes(cls, value: bytes) -> "StoredPayload":
        encoded = base64.b64encode(value).decode("ascii")
        digest, size = _inline_digest_size("base64", encoded)
        return cls("inline", "base64", digest, size, encoded)

    @classmethod
    def object(cls, reference: "ObjectRef") -> "StoredPayload":
        if not isinstance(reference, ObjectRef):
            raise TypeError("object payload reference is invalid")
        return cls("object", None, reference.digest, reference.size, ref=reference)

    def to_json(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "kind": self.kind,
            "encoding": self.encoding,
            "digest": self.digest,
            "size": self.size,
        }
        if self.kind == "inline":
            value["value"] = self.value  # type: ignore[assignment]
        else:
            assert self.ref is not None
            value["ref"] = {
                "store_id": self.ref.store_id,
                "key": self.ref.key,
            }
        return value

    @classmethod
    def from_json(cls, raw: object) -> "StoredPayload":
        if not isinstance(raw, dict):
            raise ValueError("stored payload must be an object")  # noqa: TRY004
        kind = raw.get("kind")
        encoding = raw.get("encoding")
        digest = raw.get("digest")
        size = raw.get("size")
        if (
            not isinstance(kind, str)
            or not isinstance(digest, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
        ):
            raise ValueError("stored payload fields are invalid")  # noqa: TRY004
        if kind == "inline":
            if "ref" in raw or "value" not in raw:
                raise ValueError("inline payload fields are invalid")
            return cls(kind, encoding if isinstance(encoding, str) else None, digest, size, raw["value"])
        if kind == "object":
            descriptor = raw.get("ref")
            if (
                not isinstance(descriptor, dict)
                or "value" in raw
                or encoding is not None
                or not isinstance(descriptor.get("store_id"), str)
                or not isinstance(descriptor.get("key"), str)
            ):
                raise ValueError("object payload fields are invalid")
            reference = ObjectRef(
                descriptor["store_id"],
                descriptor["key"],
                digest,
                size,
            )
            return cls(kind, None, digest, size, ref=reference)
        raise ValueError("stored payload kind is invalid")

    def decode(self) -> object:
        if self.kind != "inline":
            raise ValueError("object payload requires ObjectStore resolution")
        if self.encoding == "base64":
            return base64.b64decode(str(self.value), validate=True)
        return self.value


def payload_fits_inline(payload: StoredPayload, policy: PayloadPolicy) -> bool:
    return len(canonical_json_bytes(payload.to_json())) <= policy.inline_limit_bytes


def _inline_digest_size(encoding: str, value: object) -> tuple[str, int]:
    if encoding == "json":
        data = canonical_json_bytes(value)
    elif encoding == "utf-8":
        if not isinstance(value, str):
            raise ValueError("text payload must be a string")
        data = value.encode("utf-8")
    elif encoding == "base64":
        if not isinstance(value, str):
            raise ValueError("base64 payload must be a string")
        data = value.encode("ascii")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("base64 payload is invalid") from error
        return hashlib.sha256(decoded).hexdigest(), len(decoded)
    else:
        raise ValueError("payload encoding is invalid")
    return hashlib.sha256(data).hexdigest(), len(data)


__all__ = ["PayloadPolicy", "StoredPayload", "payload_fits_inline"]
