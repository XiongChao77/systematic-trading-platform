"""Stable execution-instance identities, independent of display names."""

from urllib.parse import quote


def instance_key(venue, account_id, symbol, interval, model_hash):
    values = (str(venue).strip().lower(), str(account_id).strip(),
              str(symbol).strip().upper(), str(interval).strip(), str(model_hash).strip())
    if any(not value for value in values):
        raise ValueError("Instance identity requires venue, account, symbol, interval and hash")
    return "|".join(quote(value, safe="") for value in values)


def display_name(symbol, interval, venue, model_hash):
    return f"{str(symbol).upper()}-{interval}-{str(venue).lower()}-{model_hash}"


def spec_for_trace(specs, trace_instance_id):
    """Match a stable captured identity without any venue discovery."""
    exact = [spec for spec in specs if spec.instance_id == trace_instance_id]
    if len(exact) != 1:
        raise ValueError(f"No unique configured instance matches trace: {trace_instance_id}")
    return exact[0]
