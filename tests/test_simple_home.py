from simple_home import recommended_strategy_id


def test_beginner_home_maps_onboarding_presets_to_plain_language_plans():
    assert recommended_strategy_id({"recommended_strategy": "Moving-average crossover"}) == "conservative_etf_growth"
    assert recommended_strategy_id({"recommended_strategy": "Day-trading AI direction model"}) == "ai_day_research"
    assert recommended_strategy_id({"recommended_strategy": "Crypto AI direction model"}) == "btc_trend_candles"


def test_beginner_home_has_a_safe_default_plan():
    assert recommended_strategy_id({}) == "conservative_etf_growth"
