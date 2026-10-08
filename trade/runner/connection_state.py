"""Shared connection gates; market-data feeds are independent of these gates."""

from trade.venue.live.ctrader.ctrader_venue import CTraderOpenApiConnection


def pipeline_connection(pipeline):
    connection = getattr(pipeline.venue, "api", None)
    return connection if isinstance(connection, CTraderOpenApiConnection) else None


def pipeline_can_trade(pipeline, received_monotonic=None):
    if not pipeline.enable:
        return False
    connection = pipeline_connection(pipeline)
    return connection is None or connection.accepts_market(received_monotonic)
