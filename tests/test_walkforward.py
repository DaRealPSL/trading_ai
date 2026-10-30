import pandas as pd

from bot.walkforward import make_folds

IDX = pd.bdate_range("2005-01-03", "2012-06-29")


def test_rolling_folds_tile_and_keep_window_length():
    folds = make_folds(IDX, train_years=5, test_years=1)
    assert len(folds) == 3
    for a, b in zip(folds, folds[1:]):
        assert b.train_end == a.test_end  # test windows are back to back
    for f in folds:
        assert f.train_start == f.train_end - pd.DateOffset(years=5)
    assert folds[0].train_start == IDX[0]
    assert folds[-1].test_end > IDX[-1]  # last fold reaches the end


def test_anchored_folds_keep_start_fixed():
    folds = make_folds(IDX, train_years=5, test_years=1, anchored=True)
    assert all(f.train_start == IDX[0] for f in folds)


def test_no_folds_when_history_too_short():
    short = pd.bdate_range("2020-01-01", "2022-12-30")
    assert make_folds(short, train_years=5) == []


LONG_IDX = pd.bdate_range("2005-01-03", "2026-10-29")


def test_cli_options_reach_the_config():
    from bot.walkforward import build_parser

    a = build_parser().parse_args(["--train-years", "3", "--test-years", "2", "--anchored"])
    assert (a.train_years, a.test_years, a.anchored) == (3, 2, True)
    d = build_parser().parse_args([])
    assert (d.train_years, d.test_years, d.anchored, d.holdout_years) == (5, 1, False, 3)


def test_each_option_really_changes_the_folds():
    first_test = {y: make_folds(LONG_IDX, train_years=y)[0].train_end for y in (3, 5, 8)}
    assert first_test[3] < first_test[5] < first_test[8]  # not identical runs
    assert len(make_folds(LONG_IDX, 5, test_years=2)) < len(make_folds(LONG_IDX, 5, test_years=1))
    anchored = make_folds(LONG_IDX, 5, 1, anchored=True)
    rolling = make_folds(LONG_IDX, 5, 1, anchored=False)
    assert len({f.train_start for f in anchored}) == 1
    assert len({f.train_start for f in rolling}) == len(rolling)