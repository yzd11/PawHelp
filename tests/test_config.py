import pytest

from app.config import Settings
from tests.fakes import fake_settings


def test_settings_repr_hides_api_key_and_database_password() -> None:
    """Catch credentials being rendered in operational logs or errors."""

    settings = fake_settings(
        openai_api_key="super-secret",
        database_url="mysql+asyncmy://pawhelp:database-secret@127.0.0.1:3306/pawhelp",
    )

    assert "super-secret" not in repr(settings)
    assert "database-secret" not in repr(settings)


@pytest.mark.parametrize("canonical,expected", [
    ("https://canonical.example/v1", "https://canonical.example/v1"),
    (None, "https://legacy.example/v1"),
])
def test_chat_base_url_prefers_canonical_name_and_keeps_openai_alias(monkeypatch, canonical, expected) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    if canonical:
        monkeypatch.setenv("CHAT_BASE_URL", canonical)
    else:
        monkeypatch.delenv("CHAT_BASE_URL", raising=False)

    settings = Settings(_env_file=None, OPENAI_MODEL="demo", OPENAI_API_KEY="secret",
                        DATABASE_URL="mysql+asyncmy://unused")

    assert str(settings.chat_base_url).rstrip("/") == expected
