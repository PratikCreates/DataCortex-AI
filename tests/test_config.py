from src.core.config import Config


def test_loads_all_numbered_blaxel_accounts(monkeypatch):
    for key_id in [1, 5]:
        monkeypatch.setenv(f"BLAXEL_ACCOUNT_{key_id}_API_KEY", f"key-{key_id}")
        monkeypatch.setenv(f"BLAXEL_ACCOUNT_{key_id}_WORKSPACE_NAME", f"workspace-{key_id}")
        monkeypatch.setenv(f"BLAXEL_ACCOUNT_{key_id}_BUDGET", "10")

    config = Config()

    loaded_ids = {account.key_id for account in config.credential_pool.accounts}
    assert {1, 5}.issubset(loaded_ids)
