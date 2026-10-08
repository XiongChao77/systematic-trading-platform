"""Order-state reduction and execution summaries without timestamp regression."""

TERMINAL_STATUSES = frozenset({"filled", "rejected", "cancelled", "expired"})
_STATE_RANK = {"unknown": 0, "submitting": 1, "submitted": 2,
               "accepted": 3, "replaced": 4, "partially_filled": 5,
               "rejected": 6, "expired": 7, "cancelled": 8, "filled": 9}


def order_status(states, fallback="unknown"):
    """Reduce observations of ONE order; cancel rejection is not order rejection.

    Terminal observations dominate delayed nonterminal observations. Actual fills
    are evaluated separately by the caller and may establish partial execution.
    """
    values = [value for value in states if value in _STATE_RANK]
    if not values:
        return fallback
    return max(values, key=lambda value: _STATE_RANK[value])


def execution_status(states, submitted_quantity, filled_quantity, fallback="submitted"):
    """Aggregate distinct child orders, retaining live children and partial fills."""
    states = list(states)
    if filled_quantity > 0:
        if submitted_quantity and filled_quantity + 1e-12 >= submitted_quantity:
            return "filled"
        return "partially_filled"
    if not states:
        return fallback
    if all(state == "filled" for state in states):
        return "filled"
    if "filled" in states or "partially_filled" in states:
        return "partially_filled"
    active = [state for state in states if state not in TERMINAL_STATUSES]
    if active:
        return order_status(active, fallback)
    if "rejected" in states:
        return "rejected"
    return order_status(states, fallback)
