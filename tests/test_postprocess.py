import pandas as pd
from src.postprocess import select_matches

def test_expected_f_examples():
    truth = {"S1-1": ["S2-1"], "S1-2": ["S2-2"], "S1-3": ["S2-3"]}
    preds = pd.DataFrame([
        ["S1-1", "S2-1", 0.6],
        ["S1-2", "S2-2", 0.4],
        ["S1-3", "S2-3", 0.99],
        ["S1-3", "S2-4", 0.70],
    ], columns=["s1_id", "cand_id", "prob"])
    out = select_matches(preds, {"method": "expected_f"})
    assert out["S1-1"] == ["S2-1"]
    assert out["S1-2"] == []
    assert out["S1-3"] == ["S2-3"]
