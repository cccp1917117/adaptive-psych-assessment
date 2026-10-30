"""Fast implementation checks, not evidence of psychological validity."""
import numpy as np
import pytest
import torch

from demo import MaskedVAE, random_mask, reconstruct, run_demo, split_indices, synthetic_responses


def test_synthetic_range_and_seed():
    a = synthetic_responses(40, 9, 5)
    assert a.shape == (40, 9) and 0 <= a.min() <= a.max() <= 4
    np.testing.assert_array_equal(a, synthetic_responses(40, 9, 5))


def test_split_has_no_participant_overlap():
    a, b, c = [set(x.tolist()) for x in split_indices(100, 42)]
    assert not (a & b or a & c or b & c)
    assert len(a | b | c) == 100


def test_random_masks_are_nontrivial():
    counts = random_mask(100, 9).sum(1)
    assert counts.min() >= 1 and counts.max() <= 8


def test_hidden_answers_do_not_leak_into_prediction():
    model = MaskedVAE(9).eval()
    a = torch.from_numpy(synthetic_responses(40, 9, 5))
    b = a.clone()
    selected = [0, 2, 4]
    hidden = [1, 3, 5, 6, 7, 8]
    b[:, hidden] = (b[:, hidden] + 1) % 5
    pa, pb = reconstruct(model, a, selected), reconstruct(model, b, selected)
    torch.testing.assert_close(pa, pb)
    torch.testing.assert_close(pa[:, selected], a[:, selected].float())


def test_end_to_end_synthetic_smoke():
    result = run_demo(n=80, items=9, short=3, epochs=2, seed=42)
    assert len(set(result["selected_item_indices_zero_based"])) == 3
    for method in result["test_metrics"].values():
        assert all(np.isfinite(v) and v >= 0 for v in method.values())
    assert "SYNTHETIC" in result["status"]


@pytest.mark.parametrize("short", [0, 9, 10])
def test_invalid_short_length(short):
    with pytest.raises(ValueError):
        run_demo(n=40, items=9, short=short, epochs=1)
