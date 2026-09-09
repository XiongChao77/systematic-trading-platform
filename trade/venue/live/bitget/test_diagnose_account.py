"""Offline safety and interpretation tests for Bitget account diagnostics."""

import base64
import hashlib
import hmac
import json
from unittest.mock import Mock

import pytest
import requests

from trade.venue.live.bitget.diagnose_account import (
    ACCOUNT_INFO,
    CONTRACTS,
    DiagnosticClient,
    diagnose,
    select_targets,
)
from trade.venue.live.bitget import diagnose_account


@pytest.fixture
def client(tmp_path):
    for name, value in {
        "apikey": "test-api-key-value",
        "secret_key": "test-signing-secret-value",
        "passphrase": "test-passphrase-value",
    }.items():
        (tmp_path / name).write_text(value)
    session = Mock()
    return DiagnosticClient(tmp_path, session=session)


def response(data=None, code="00000", msg="success"):
    return Mock(
        status_code=200,
        json=Mock(
            return_value={"code": code, "msg": msg, "data": data, "requestTime": 1}
        ),
    )


def test_get_signs_the_exact_query_and_blocks_other_endpoints(client):
    client.session.get.return_value = response({})
    client.get(ACCOUNT_INFO, {"example": "a b"})
    call = client.session.get.call_args
    headers = call.kwargs["headers"]
    payload = headers["ACCESS-TIMESTAMP"] + "GET" + ACCOUNT_INFO + "?example=a+b"
    expected = base64.b64encode(
        hmac.new(client.secrets[1].encode(), payload.encode(), hashlib.sha256).digest()
    ).decode()
    assert headers["ACCESS-SIGN"] == expected
    assert call.kwargs["allow_redirects"] is False
    with pytest.raises(ValueError, match="not allowed"):
        client.get("/api/v2/mix/order/place-order")
    client.session.get.assert_called_once()
    client.get(CONTRACTS)
    assert "ACCESS-KEY" not in client.session.get.call_args.kwargs["headers"]
    client.session.post.assert_not_called()


@pytest.mark.parametrize("permissions", [["coow", "cpow"], ["coor", "cpor"], None])
def test_permissions_do_not_prove_trading_access(client, permissions):
    client.session.get.side_effect = [
        response(
            {
                "userId": "123456789",
                "parentId": 1234,
                "authorities": permissions,
                "ips": "192.0.2.1",
            }
        ),
        response([{"symbol": "DOGEUSDT", "symbolStatus": "normal"}]),
        response(
            {"posMode": "one_way_mode", "marginMode": "crossed", "accountEquity": "999"}
        ),
    ]
    result = diagnose(client, "Doge-bitget-1", "DOGEUSDT")
    assert result["checks_passed"]
    assert result["trading_access"] == "unverified"
    assert result["order_submission_tested"] is False
    details = result["checks"][0]["details"]
    if permissions is None:
        assert "futures_order_write_permission" not in details
    else:
        assert details["futures_order_write_permission"] == ("coow" in permissions)
    encoded = json.dumps(result)
    assert "123456789" not in encoded
    assert "192.0.2.1" not in encoded
    assert "accountEquity" not in encoded
    for secret in client.secrets:
        assert secret not in encoded


def test_account_rejection_does_not_abort_other_checks(client):
    client.session.get.side_effect = [
        response(code="40022", msg="restricted " + client.secrets[0]),
        response([{"symbol": "DOGEUSDT", "symbolStatus": "restrictedAPI"}]),
        response({"posMode": "hedge_mode"}),
    ]
    result = diagnose(client, "failed-strategy", "DOGEUSDT")
    assert len(result["checks"]) == 3
    assert result["checks"][0]["code"] == "40022"
    assert client.secrets[0] not in json.dumps(result)
    assert "contract_status_not_normal" in result["findings"]
    assert not result["checks_passed"]


def test_transport_error_does_not_print_credentials(client):
    client.session.get.side_effect = requests.ConnectionError(
        "headers=" + repr(client.secrets)
    )
    check, data = client.get(ACCOUNT_INFO)
    assert check["error_type"] == "ConnectionError"
    assert data is None
    for secret in client.secrets:
        assert secret not in json.dumps(check)


def test_live_config_resolves_selected_strategy_keys(tmp_path):
    path = tmp_path / "live_config.json"
    path.write_text(
        json.dumps(
            {
                "strategy": {
                    "Doge-bitget-1": {
                        "venue": "bitget",
                        "bitget": {"path": "bitget/trading7"},
                    },
                    "Other": {"venue": "binance"},
                }
            }
        )
    )
    assert select_targets(path, ["Doge-bitget-1", "Doge-bitget-1"]) == [
        ("Doge-bitget-1", tmp_path / "bitget/trading7")
    ]
    with pytest.raises(ValueError, match="does not use Bitget"):
        select_targets(path, ["Other"])


def test_cli_writes_private_report_without_exposing_secrets(
    client, tmp_path, monkeypatch, capsys
):
    client.session.get.side_effect = [
        response({"authorities": ["coow"]}),
        response([{"symbol": "DOGEUSDT", "symbolStatus": "normal"}]),
        response({"posMode": "hedge_mode"}),
    ]
    output = tmp_path / "report.json"
    monkeypatch.setattr(diagnose_account, "DiagnosticClient", lambda path: client)
    monkeypatch.setattr(diagnose_account.time, "sleep", lambda delay: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_account",
            "--key-path",
            str(tmp_path),
            "--strategy-id",
            "Doge-bitget-1",
            "--output",
            str(output),
        ],
    )
    assert diagnose_account.main() == 0
    report = json.loads(output.read_text())
    assert report["accounts"][0]["strategy_id"] == "Doge-bitget-1"
    assert output.stat().st_mode & 0o777 == 0o600
    client.session.close.assert_called_once()
    rendered = capsys.readouterr().out + output.read_text()
    for secret in client.redactions:
        assert secret not in rendered
