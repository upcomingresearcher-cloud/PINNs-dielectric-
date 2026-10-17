"""Shared pytest configuration and helpers."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def reproduction_tolerance(stored_hardware):
    """Relative tolerance for re-training a stored run.

    On the CPU model that produced the stored row, results must agree to round-off (1e-12).
    On a different CPU model, XLA generates different machine code and the last digits of
    floating-point results can differ, so agreement is required to 1e-6 relative instead.
    """
    from src.experiments.provenance import hardware
    return 1e-12 if hardware() == stored_hardware else 1e-6
