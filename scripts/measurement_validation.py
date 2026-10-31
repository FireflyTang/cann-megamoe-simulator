"""Strict CPU admission for tolerances and SYS_CNT frequency (MHz)."""
import math


def _finite_number(value, field, *, positive):
    # bool subclasses int, and float("nan")/float("inf") are valid JSON5 values.
    # Do not coerce strings or alternate units at this contract boundary.
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite numeric value (not bool or string)")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{field} must be finite and {bound}")
    return value


def validate_tolerances(validation):
    if not isinstance(validation, dict):
        raise ValueError("validation must be an object with explicit rtol and atol")
    for key in ("rtol", "atol"):
        if key not in validation:
            raise ValueError(f"validation.{key} is required")
        _finite_number(validation[key], f"validation.{key}", positive=False)


def validate_frequency_mhz(value):
    """Return a checked number; cycles / MHz is microseconds, not core clock Hz."""
    return _finite_number(value, "profiling.timestamp_frequency_mhz", positive=True)


def validate_measurement_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    validate_tolerances(config.get("validation"))
    profiling = config.get("profiling")
    if not isinstance(profiling, dict) or "timestamp_frequency_mhz" not in profiling:
        raise ValueError("profiling.timestamp_frequency_mhz is required (MHz)")
    validate_frequency_mhz(profiling["timestamp_frequency_mhz"])
