"""
GeoJSON layer persistence for the manual digitizer.

Each layer is stored as a separate FeatureCollection JSON file inside
{output_dir}/digitizer/  so it never touches the main pipeline outputs
until the user explicitly clicks Export.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.manual_digitizer.config import LAYERS


class GeoJSONStore:
    """Read/write per-layer GeoJSON files in the digitizer data directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.store_dir  = self.output_dir / "digitizer"
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Initialise empty layers on first run
        for layer in LAYERS:
            path = self._path(layer)
            if not path.exists():
                self._write(layer, [])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _path(self, layer: str) -> Path:
        return self.store_dir / f"{layer}.geojson"

    def _write(self, layer: str, features: List[dict]) -> None:
        fc = {"type": "FeatureCollection", "features": features}
        self._path(layer).write_text(json.dumps(fc, indent=2), encoding="utf-8")

    def _read(self, layer: str) -> dict:
        path = self._path(layer)
        if not path.exists():
            return {"type": "FeatureCollection", "features": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"type": "FeatureCollection", "features": []}

    @staticmethod
    def _ensure_id(feature: dict) -> str:
        """Ensure feature has a stable UUID id in properties. Returns the id."""
        props = feature.setdefault("properties", {})
        if not props.get("id"):
            props["id"] = str(uuid.uuid4())
        return props["id"]

    # ── Public API ────────────────────────────────────────────────────────────

    def get_all(self) -> Dict[str, dict]:
        """Return all layers as {layerName: FeatureCollection}."""
        return {layer: self._read(layer) for layer in LAYERS}

    def get_layer(self, layer: str) -> dict:
        return self._read(layer)

    def save_layer(self, layer: str, features: List[dict]) -> int:
        """Replace an entire layer atomically. Returns new feature count."""
        for f in features:
            self._ensure_id(f)
        self._write(layer, features)
        return len(features)

    def upsert_feature(self, layer: str, feature: dict) -> str:
        """Insert or update a single feature by id. Returns the feature id."""
        fc       = self._read(layer)
        features = fc["features"]
        fid      = self._ensure_id(feature)

        for i, f in enumerate(features):
            if f.get("properties", {}).get("id") == fid:
                features[i] = feature
                self._write(layer, features)
                return fid

        features.append(feature)
        self._write(layer, features)
        return fid

    def delete_feature(self, layer: str, feature_id: str) -> bool:
        """Remove feature by id. Returns True if the feature was found."""
        fc       = self._read(layer)
        features = fc["features"]
        kept     = [f for f in features if f.get("properties", {}).get("id") != feature_id]
        if len(kept) == len(features):
            return False
        self._write(layer, kept)
        return True

    def feature_counts(self) -> Dict[str, int]:
        """Return {layerName: featureCount} for all layers."""
        return {layer: len(self._read(layer)["features"]) for layer in LAYERS}

    def clear_layer(self, layer: str) -> None:
        self._write(layer, [])
