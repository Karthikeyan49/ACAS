"""
core/config_loader.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Single shared loader for config/thresholds.yaml. Every module that needs
    an operational threshold reads it through here, so the YAML file is the
    one real source of truth (CONTRIBUTING.md: "Edit thresholds in
    config/thresholds.yaml, not source code").

CALLED FROM
    core/risk_scorer.py     alert bands, fuel scaling, limitation constants
    core/decision_fsm.py    decision-gate (ground-veto) parameters
    core/pc_analytic.py     hard-body radius
    api/endpoint.py         /api/thresholds response

CALLS INTO
    Standard library only (PyYAML used when present, with a built-in
    minimal parser as fallback so flight software needs no extra package).

WHAT IT PROVIDES
    load_config(path=None)  → dict     parsed thresholds.yaml, cached
    get(key, default=None)  → value    dotted-path lookup, e.g.
                                        get("alerts.red"), get("screening_km")
    reload()                           drop the cache (tests / hot reload)
══════════════════════════════════════════════════════════════════════════════
"""
import os

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "thresholds.yaml",
)

_cache = {}


def _parse_scalar(text: str):
    text = text.strip()
    if not text:
        return None
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            pass
    return text.strip("'\"")


def _minimal_yaml(text: str) -> dict:
    """Parse the flat / one-level-nested subset of YAML used by
    thresholds.yaml without any third-party dependency."""
    root, current = {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        key, value = key.strip(), value.strip()
        if indent == 0:
            if value:
                root[key] = _parse_scalar(value)
                current = None
            else:
                root[key] = {}
                current = root[key]
        elif current is not None and value:
            current[key] = _parse_scalar(value)
    return root


def load_config(path: str = None) -> dict:
    path = path or _CONFIG_PATH
    if path in _cache:
        return _cache[path]
    with open(path) as f:
        text = f.read()
    try:
        import yaml
        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = _minimal_yaml(text)
    _cache[path] = cfg or {}
    return _cache[path]


def get(dotted_key: str, default=None, path: str = None):
    node = load_config(path)
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def reload():
    _cache.clear()
