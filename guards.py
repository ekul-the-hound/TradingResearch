# ==============================================================================
# guards.py -- runtime preconditions that keep this build local-only
# ==============================================================================
# This is the leak a "grep for claude" audit misses: discovery's "cloud" and
# "hybrid" modes route to Ollama Cloud (:cloud model tags, `ollama signin`),
# which is external and non-free even though it is not Anthropic.
#
# assert_local_only() is called at the very top of run_pipeline.py and
# discovery_pipeline.py, before any model work begins. It refuses to start if
# the resolved discovery mode or any resolved model tag would touch the cloud.
# ==============================================================================

# Modes that are cloud-backed and therefore forbidden in a local-only build.
_CLOUD_MODES = {"cloud", "hybrid"}

# Substrings / exact tags known to resolve to Ollama Cloud GPUs.
_CLOUD_TAG_SUBSTRINGS = (":cloud",)
_KNOWN_CLOUD_TAGS = {
    "qwen3-coder:480b-cloud",
    "qwen3.5:cloud",
    "minimax-m3:cloud",
}


def _tag_is_cloud(tag: str) -> bool:
    if not tag:
        return False
    t = str(tag).strip().lower()
    if t in {x.lower() for x in _KNOWN_CLOUD_TAGS}:
        return True
    return any(sub in t for sub in _CLOUD_TAG_SUBSTRINGS)


def assert_local_only() -> None:
    """
    Refuse to start unless discovery is in a local mode AND every model tag
    that will actually be used is a non-cloud tag. Raises RuntimeError naming
    the offending mode/tag. Safe to call even if discovery_config can't import.
    """
    try:
        import discovery_config as dc
    except Exception as e:  # pragma: no cover - config must import to run at all
        raise RuntimeError(
            f"assert_local_only: cannot import discovery_config to verify "
            f"local-only status ({type(e).__name__}: {e})"
        )

    mode = getattr(dc, "MODE", None)

    if mode in _CLOUD_MODES:
        raise RuntimeError(
            f"This build is local-only. DISCOVERY_MODE='{mode}' is a cloud "
            f"mode. Set DISCOVERY_MODE=local27 and use non-:cloud model tags."
        )

    # Resolve every model tag this mode will actually use, across all roles.
    active = dc.MODES.get(mode)
    if active is None:
        raise RuntimeError(
            f"This build is local-only. DISCOVERY_MODE='{mode}' is not a known "
            f"mode. Set DISCOVERY_MODE=local27 and use non-:cloud model tags."
        )

    role_keys = ("summarizer_model", "code_model", "reviewer_model")
    offenders = []
    for key in role_keys:
        tag = active.get(key, "")
        if _tag_is_cloud(tag):
            offenders.append(f"{key}={tag}")

    if offenders:
        raise RuntimeError(
            "This build is local-only. Cloud model tag(s) detected for "
            f"DISCOVERY_MODE='{mode}': {', '.join(offenders)}. "
            "Set DISCOVERY_MODE=local27 and use non-:cloud model tags."
        )


if __name__ == "__main__":
    try:
        assert_local_only()
        import discovery_config as dc
        print(f"[OK] Local-only guard passed. DISCOVERY_MODE={dc.MODE}")
        a = dc.MODES[dc.MODE]
        print(f"     summarizer = {a['summarizer_model']}")
        print(f"     code_model = {a['code_model']}")
        print(f"     reviewer   = {a['reviewer_model']}")
    except RuntimeError as e:
        print(f"[BLOCKED] {e}")
        raise SystemExit(1)
