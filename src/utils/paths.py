from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[2]

def research_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else RESEARCH_ROOT / path

def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
