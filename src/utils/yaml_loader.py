"""Lightweight YAML parser with PyYAML preference and pure-Python fallback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file using PyYAML if available, else standard fallback parser."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    # Pure Python basic YAML dictionary parser for key-value and nested dicts
    result: dict[str, Any] = {}
    current_section: str | None = None
    
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
            
        # Check indentation for nested mapping
        indent = len(line) - len(line.lstrip())
        
        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            
            # Remove inline comments
            if "#" in val:
                val = val.split("#", 1)[0].strip()
                
            if not val:
                current_section = key
                result[current_section] = {}
                continue
                
            # Parse primitive types
            parsed_val: Any
            if val.lower() == "true":
                parsed_val = True
            elif val.lower() == "false":
                parsed_val = False
            elif val.lower() in ("null", "none", "~"):
                parsed_val = None
            elif val.startswith("[") and val.endswith("]"):
                items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
                parsed_val = items
            elif val.startswith("{") and val.endswith("}"):
                parsed_val = {}
            else:
                try:
                    parsed_val = int(val)
                except ValueError:
                    try:
                        parsed_val = float(val)
                    except ValueError:
                        parsed_val = val.strip("'\"")
                        
            if indent > 0 and current_section:
                result[current_section][key] = parsed_val
            else:
                current_section = None
                result[key] = parsed_val
                
    return result


def dump_yaml(data: dict[str, Any]) -> str:
    """Dump dictionary to YAML format."""
    try:
        import yaml
        return yaml.safe_dump(data, sort_keys=False)
    except ImportError:
        pass
        
    lines = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for sub_k, sub_v in v.items():
                lines.append(f"  {sub_k}: {sub_v}")
        elif isinstance(v, list):
            lines.append(f"{k}: {json.dumps(v)}")
        elif v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: '{v}'")
    return "\n".join(lines) + "\n"
