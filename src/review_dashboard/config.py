"""Configuration and path layout for the blind human review dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def default_repo_root() -> Path:
    """Resolve repository root relative to this module."""
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class DashboardConfig:
    """Runtime configuration for the blind human GT v1.1 review server."""

    repo_root: Path
    data_root: Path
    v1_1_release_dir: Path
    workqueue_csv: Path
    release_metadata_json: Path
    release_schema_json: Path
    sync_metadata_json: Path
    pair_manifest_csv: Path
    workspace_dir: Path
    db_path: Path
    host: str = "127.0.0.1"
    port: int = 8765
    allow_proxy_transcode: bool = True
    max_pts_index_pairs: int = 2
    max_proxy_pairs: int = 2

    @classmethod
    def from_repo(
        cls,
        repo_root: Path | None = None,
        *,
        workspace_dir: Path | None = None,
        db_path: Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        allow_proxy_transcode: bool = True,
    ) -> DashboardConfig:
        """Build default paths anchored inside the repository."""
        root = (repo_root or default_repo_root()).resolve()
        data_root = root / "data" / "dual_camera_test_v1"
        v1_1_dir = data_root / "releases" / "dual_camera_gt_v1.1_blind_human"
        ws_dir = Path(
            workspace_dir
            or os.environ.get("GT_REVIEW_WORKSPACE_DIR")
            or (v1_1_dir / "review_workspace")
        ).resolve()
        resolved_db = Path(
            db_path
            or os.environ.get("GT_REVIEW_DB_PATH")
            or (ws_dir / "review_dashboard.sqlite3")
        ).resolve()

        return cls(
            repo_root=root,
            data_root=data_root,
            v1_1_release_dir=v1_1_dir,
            workqueue_csv=v1_1_dir / "blind_annotation_workqueue.csv",
            release_metadata_json=v1_1_dir / "release_metadata.json",
            release_schema_json=v1_1_dir / "schema.json",
            sync_metadata_json=data_root / "metadata" / "synchronization_metadata.json",
            pair_manifest_csv=data_root / "metadata" / "pair_manifest.csv",
            workspace_dir=ws_dir,
            db_path=resolved_db,
            host=host,
            port=int(port),
            allow_proxy_transcode=allow_proxy_transcode,
        )

    @property
    def submissions_dir(self) -> Path:
        return self.workspace_dir / "submissions"

    @property
    def adjudications_dir(self) -> Path:
        return self.workspace_dir / "adjudications"

    @property
    def snapshots_dir(self) -> Path:
        return self.workspace_dir / "snapshots"

    @property
    def exports_dir(self) -> Path:
        return self.workspace_dir / "exports"

    @property
    def backups_dir(self) -> Path:
        return self.workspace_dir / "backups"

    @property
    def proxy_cache_dir(self) -> Path:
        return self.workspace_dir / ".proxy_cache"

    def ensure_workspace_dirs(self) -> None:
        """Create isolated review workspace subdirectories without touching v1.0 or root manifests."""
        for p in (
            self.workspace_dir,
            self.db_path.parent,
            self.submissions_dir / "reviewer_a",
            self.submissions_dir / "reviewer_b",
            self.adjudications_dir,
            self.snapshots_dir / "reviewer_a",
            self.snapshots_dir / "reviewer_b",
            self.exports_dir,
            self.backups_dir,
            self.proxy_cache_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
