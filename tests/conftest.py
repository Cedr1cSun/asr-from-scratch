def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: builds real models / trainers; deselect with -m 'not slow'"
    )
