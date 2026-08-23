"""Production principal provider adapter."""

from ..core import Principal


class StaticPrincipalProvider:
    def __init__(self, principal: Principal) -> None:
        self._principal = principal

    async def current(self) -> Principal:
        return self._principal


__all__ = ["StaticPrincipalProvider"]
