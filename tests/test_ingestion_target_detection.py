import pandas as pd

from src.agents.ingestion_agent import IngestionAgent
from src.core.config import Config


class DummyLLM:
    pass


def test_market_basket_columns_do_not_auto_select_yams_as_target():
    df = pd.DataFrame(
        {
            "shrimp": ["mineral water", "burgers", "turkey", "eggs"],
            "almonds": ["spaghetti", None, "milk", None],
            "avocado": [None, "eggs", None, None],
            "yams": [None, "green tea", None, None],
            "spinach": [None, None, None, "cereals"],
        }
    )
    agent = IngestionAgent(Config(), DummyLLM())
    schema = agent._detect_schema(df)

    assert agent._guess_target_column(df, schema) is None


def test_market_basket_cleaning_preserves_duplicate_transactions():
    df = pd.DataFrame(
        {
            "Item 1": ["whole milk", "whole milk", "soda"],
            "Item 2": ["bread", "bread", None],
            "Item 3": [None, None, None],
            "Item 4": [None, None, None],
        }
    )
    agent = IngestionAgent(Config(), DummyLLM())

    cleaned, log = agent._auto_clean(df)

    assert len(cleaned) == 3
    assert any("Kept 1 repeated basket rows" in action for action in log["actions"])


def test_exact_y_column_can_still_be_target():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "y": [0, 1, 0, 1],
        }
    )
    agent = IngestionAgent(Config(), DummyLLM())
    schema = agent._detect_schema(df)

    assert agent._guess_target_column(df, schema) == "y"
