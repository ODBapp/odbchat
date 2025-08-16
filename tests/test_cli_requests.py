import pytest
from cli.mhw_cli_patch import _parse_mhw_flags, _canonical_fetch_arglist
from shared.schemas import canonicalize_fetch, validate_fetch

def test_build_requests_multip_periods_antimeridian():
    line_tokens = [
        "--bbox","135,-25,-60,25",
        "--periods","202001-202003,202101",
        "--fields","sst_anomaly"
    ]
    p = _parse_mhw_flags(line_tokens)
    arglist = _canonical_fetch_arglist(p)
    # anti-meridian → should split into 2 bboxes per period → total 3 requests x 2 = 6
    assert len(arglist) == 6
    for a in arglist:
        a1 = canonicalize_fetch(a)
        ok, err = validate_fetch(a1)
        assert ok, err