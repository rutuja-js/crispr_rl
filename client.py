"""Python client wrapper for the crispr_rl FastAPI service."""

from __future__ import annotations

from typing import Any

import httpx


class CRISPRClient:
    """Async + sync Python client for the crispr_rl REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Sync interface (thin wrappers)
    # ------------------------------------------------------------------

    def ping(self) -> dict:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.base_url}/crispr/ping")
            r.raise_for_status()
            return r.json()

    def design(
        self,
        sequence: str | None = None,
        gene_ids: list[str] | None = None,
        profile: str = "knockout",
        seed: int = 42,
        params: dict | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "gene_ids": gene_ids or [],
            "profile": profile,
            "seed": seed,
            "params": params or {},
        }
        if sequence:
            payload["sequence"] = sequence
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.base_url}/crispr/design", json=payload)
            r.raise_for_status()
            return r.json()

    def feedback(self, candidate_id: str, rating: int, notes: str = "", rationale: str = "") -> dict:
        payload = {"candidate_id": candidate_id, "rating": rating, "notes": notes, "rationale": rationale}
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.base_url}/crispr/feedback", json=payload)
            r.raise_for_status()
            return r.json()

    def get_config(self) -> dict:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.base_url}/crispr/config")
            r.raise_for_status()
            return r.json()

    def update_config(self, profile: str | None = None, weights: dict | None = None) -> dict:
        payload: dict[str, Any] = {}
        if profile:
            payload["profile"] = profile
        if weights:
            payload["weights"] = weights
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.base_url}/crispr/config", json=payload)
            r.raise_for_status()
            return r.json()

    def get_metrics(self) -> dict:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.base_url}/metrics")
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def async_design(self, **kwargs) -> dict:
        payload: dict[str, Any] = {
            "gene_ids": kwargs.get("gene_ids", []),
            "profile": kwargs.get("profile", "knockout"),
            "seed": kwargs.get("seed", 42),
            "params": kwargs.get("params", {}),
        }
        if kwargs.get("sequence"):
            payload["sequence"] = kwargs["sequence"]
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/crispr/design", json=payload)
            r.raise_for_status()
            return r.json()

    async def async_feedback(self, candidate_id: str, rating: int, **kwargs) -> dict:
        payload = {"candidate_id": candidate_id, "rating": rating, **kwargs}
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/crispr/feedback", json=payload)
            r.raise_for_status()
            return r.json()
