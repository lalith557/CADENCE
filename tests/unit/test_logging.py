from cadence.common.logging import get_logger


def test_get_logger_returns_bound_logger():
    log = get_logger("test")
    log.info("hello", answer=42)
