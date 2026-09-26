from src.metrics import blocking_report, f05_entity, macro_f05


def test_problem_statement_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert round(f05_entity(pred, true), 3) == 0.714


def test_empty_cases():
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity(set(), {"S2-1"}) == 0.0
    assert f05_entity({"S2-1"}, set()) == 0.0


def test_perfect_prediction():
    assert f05_entity({"S2-1", "S3-1"}, {"S2-1", "S3-1"}) == 1.0
    assert macro_f05({"S1-1": ["S2-1"]}, {"S1-1": ["S2-1"]}) == 1.0


def test_macro_missing_prediction_counts_as_empty():
    score, per_entity = macro_f05(
        {"S1-1": ["S2-1"]},
        {"S1-1": ["S2-1"], "S1-2": []},
        return_per_entity=True,
    )
    assert score == 1.0
    assert per_entity.loc["S1-1"] == 1.0
    assert per_entity.loc["S1-2"] == 1.0


def test_blocking_report():
    candidates = __import__("pandas").DataFrame(
        {"s1_id": ["S1-1", "S1-1", "S1-2"], "cand_id": ["S2-1", "S3-1", "S2-2"]}
    )
    truth = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-3"], "S1-3": []}
    report = blocking_report(candidates, truth)
    assert report["pair_recall"] == 2 / 3
    assert report["entity_ceiling"] == 2 / 3
    assert report["oracle_f05"] == 2 / 3
    assert report["mean_candidates"] == 1.0
