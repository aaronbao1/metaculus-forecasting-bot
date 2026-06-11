from scripts.fit_calibration import join_resolutions


def test_join_resolutions_matches_on_question_id():
    logs = [{"question_id": 1, "type": "binary", "final": 0.8},
            {"question_id": 2, "type": "binary", "final": 0.3},
            {"question_id": 3, "type": "numeric", "final": None}]
    resolutions = {1: 1, 2: 0}            # question_id -> outcome (1=yes)
    preds, outcomes = join_resolutions(logs, resolutions)
    assert preds == [0.8, 0.3]
    assert outcomes == [1, 0]
