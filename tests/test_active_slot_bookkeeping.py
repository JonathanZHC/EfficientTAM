"""CPU-only checks of the active-slot bookkeeping of fixed-batch propagation (no model weights).

Run: python -m pytest tests/test_active_slot_bookkeeping.py
"""
from types import SimpleNamespace

import pytest
import torch

from efficient_track_anything.efficienttam_video_predictor import EfficientTAMVideoPredictor as P


def _predictor(views, slots, skip=True):
    p = SimpleNamespace(
        fixed_num_views=views,
        max_objects_per_view=slots,
        fixed_batch_skip_inactive=skip,
        _active_index_cache={},
        _no_obj_rows={},
    )
    for name in ("fixed_batch_buckets", "set_fixed_batch_active_slots", "_fixed_batch_active_order",
                 "_fixed_batch_slot_order", "_active_row_index", "_no_obj_row", "_get_obj_num"):
        setattr(p, name, getattr(P, name).__get__(p))
    return p


def _states(views, slots):
    return [{"obj_idx_to_id": {i: i for i in range(slots)}} for _ in range(views)]


@pytest.mark.parametrize("views,slots,expected", [(1, 1, [1]), (2, 1, [1, 2]), (2, 4, [1, 2, 4, 8]),
                                                  (2, 8, [1, 2, 4, 8, 16]), (2, 3, [1, 2, 4, 6])])
def test_buckets(views, slots, expected):
    assert _predictor(views, slots).fixed_batch_buckets() == expected


def test_active_order_and_rows():
    p = _predictor(2, 4)
    states = _states(2, 4)
    assert p._fixed_batch_active_order(states) == [(v, o) for v in range(2) for o in range(4)]   # default: all
    p.set_fixed_batch_active_slots(states, [[2, 0], []])
    active = p._fixed_batch_active_order(states)
    assert active == [(0, 0), (0, 2)]
    # rows into [active..., NO_OBJ]: view 0 slots 0 and 2 from the batch, the rest from the NO_OBJ row
    assert p._active_row_index(active, 0, "cpu").tolist() == [0, 2, 1, 2]
    assert p._active_row_index(active, 1, "cpu").tolist() == [2, 2, 2, 2]
    with pytest.raises(ValueError):
        p.set_fixed_batch_active_slots(states, [[4], []])


def test_skipping_off_ignores_active_marks():
    p = _predictor(2, 2, skip=False)
    states = _states(2, 2)
    p.set_fixed_batch_active_slots(states, [[0], []])
    assert len(p._fixed_batch_active_order(states)) == 4


def test_no_obj_row_is_cached():
    p = _predictor(1, 1)
    like = torch.zeros(1, dtype=torch.float32)
    row = p._no_obj_row(like, 6, 9)
    assert row.shape == (1, 1, 6, 9) and float(row.max()) == -1024.0
    assert p._no_obj_row(like, 6, 9) is row
