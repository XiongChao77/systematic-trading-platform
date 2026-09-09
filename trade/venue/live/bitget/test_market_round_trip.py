"""Offline market round-trip tests using the stateful exchange simulator."""

import pytest

from trade.venue.live.bitget.market_round_trip import round_trip
from trade.venue.live.bitget.test_bitget_venue import (
    Session,
    credentials,
    venue_factory,
)


def small_lot_session(hedge=False):
    session = Session(hedge=hedge)
    session.contract.update(sizeMultiplier="0.00001", minTradeNum="0.00001")
    return session


@pytest.mark.parametrize("hedge", [False, True])
def test_market_entry_is_closed_and_flat_is_verified(venue_factory, hedge):
    session = small_lot_session(hedge)
    venue = venue_factory(session=session, read_only=True)
    events = []
    result = round_trip(
        venue, lambda event, data: events.append((event, data)), execute=True
    )
    assert result == {"status": "completed", "flat": True, "entry_error": None}
    posts = [c for c in session.calls if c[0] == "POST"]
    assert len(posts) == 2
    assert all(c[2]["orderType"] == "market" for c in posts)
    assert float(posts[0][2]["size"]) * 60001 <= 10
    assert (
        posts[1][2].get("tradeSide") == "close"
        if hedge
        else posts[1][2]["reduceOnly"] == "YES"
    )
    assert not session.positions
    assert any(event == "close_result" for event, _ in events)


@pytest.mark.parametrize("failure", ["reject_entry", "timeout_after_entry"])
def test_entry_failure_never_retries_entry_and_leaves_flat(venue_factory, failure):
    session = small_lot_session()
    setattr(session, failure, True)
    venue = venue_factory(session=session, read_only=True)
    result = round_trip(venue, lambda *args: None, execute=True)
    assert result["status"] == "entry_failed"
    assert result["flat"]
    entries = [
        c
        for c in session.calls
        if c[0] == "POST"
        and c[1].endswith("place-order")
        and c[2].get("reduceOnly") != "YES"
    ]
    assert len(entries) == 1
    assert not session.positions


def test_preexisting_position_blocks_all_mutations(venue_factory):
    session = small_lot_session()
    session.positions = [session.position()]
    venue = venue_factory(session=session, read_only=True)
    with pytest.raises(RuntimeError, match="existing position"):
        round_trip(venue, lambda *args: None, execute=True)
    assert not [c for c in session.calls if c[0] == "POST"]


def test_default_is_read_only(venue_factory):
    session = small_lot_session()
    venue = venue_factory(session=session, read_only=True)
    assert round_trip(venue, lambda *args: None)["status"] == "preflight_only"
    assert not [c for c in session.calls if c[0] == "POST"]
