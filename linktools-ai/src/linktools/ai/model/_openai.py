"""OpenAI model binding."""

from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from linktools.core import environ
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..core import canonical_sha256

_logger = environ.get_logger("ai.model.openai")


@dataclass(frozen=True, slots=True)
class _OpenAIModelBinding:
    route_id: str
    model: str
    base_url: "str | None" = None
    api_key: "str | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.route_id.strip() or not self.model.strip():
            raise ValueError("OpenAI model binding is incomplete")
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))
        if self.api_key is not None and not self.api_key.strip():
            object.__setattr__(self, "api_key", None)

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_identity(self) -> str:
        return f"openai:{self.model}"

    @property
    def connection_identity(self) -> str:
        credential_id = "openai-api-key" if self.api_key is not None else None
        return canonical_sha256({"provider": self.provider, "base_url": self.base_url, "credential_id": credential_id})

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"route_id": self.route_id, "provider": self.provider, "model_identity": self.model_identity, "connection_identity": self.connection_identity})

    def materialize(self) -> Model:
        model = OpenAIChatModel(self.model.removeprefix("openai:"), provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key))
        _logger.debug("OpenAI model materialized: route=%s model=%s credential=%s", self.route_id, self.model, self.api_key is not None)
        return model


def _normalize_base_url(value: "str | None") -> "str | None":
    if value is None or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("model base_url port is invalid") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("model base_url must be a clean absolute HTTP URL")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return urlunsplit((parsed.scheme.lower(), host if port is None else f"{host}:{port}", parsed.path.rstrip("/"), "", ""))


__all__ = ["_OpenAIModelBinding"]
