"""Text preprocessing pipeline used at training and inference time.

The full implementation lands in S1-T6. This stub exists so the package
imports cleanly during the scaffolding sprint task.
"""

__all__ = ["clean_text", "build_numeric_features"]


def clean_text(text: str) -> str:
    """Placeholder — full pipeline implemented in S1-T6."""
    raise NotImplementedError("clean_text() will be implemented in S1-T6.")


def build_numeric_features(text: str) -> dict[str, int]:
    """Placeholder — full pipeline implemented in S1-T6."""
    raise NotImplementedError("build_numeric_features() will be implemented in S1-T6.")
