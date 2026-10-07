"""`ScenePaddedBatchSampler` bounds the padded cost `width x sum(tracks)` of a batch."""

import pytest

from cast_asd.data.samplers import ScenePaddedBatchSampler


def padded_cost(batch, frames, tracks):
    return max(frames[i] for i in batch) * sum(tracks[i] for i in batch)


def test_the_padded_bound_is_never_exceeded():
    # Equal products, opposite shapes: bucketing on frames x tracks would fail here.
    frames = [750, 250, 750, 250, 500, 500] * 20
    tracks = [2, 6, 2, 6, 3, 3] * 20
    cap = 6000
    sampler = ScenePaddedBatchSampler(
        frames, tracks, frames_per_batch=cap, max_batch_size=32, shuffle=False
    )
    assert sorted(i for b in sampler for i in b) == list(range(len(frames)))
    for batch in sampler:
        assert padded_cost(batch, frames, tracks) <= cap


def test_a_scene_larger_than_the_budget_travels_alone():
    frames = [750, 100, 100]
    tracks = [9, 1, 1]
    sampler = ScenePaddedBatchSampler(
        frames, tracks, frames_per_batch=2000, max_batch_size=32, shuffle=False
    )
    batches = list(sampler)
    assert [0] in batches
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_max_batch_size_is_honoured():
    sampler = ScenePaddedBatchSampler(
        [10] * 50, [1] * 50, frames_per_batch=10_000, max_batch_size=4, shuffle=False
    )
    assert all(len(b) <= 4 for b in sampler)


def test_shuffling_changes_the_grouping_across_epochs():
    sampler = ScenePaddedBatchSampler(
        [750] * 60, [2] * 60, frames_per_batch=6000, max_batch_size=32, shuffle=True
    )
    first = [list(b) for b in sampler]
    sampler.set_epoch(1)
    assert [list(b) for b in sampler] != first


@pytest.mark.parametrize("cap", [1500, 6000, 24000])
def test_every_scene_appears_exactly_once_at_any_budget(cap):
    frames = [750, 613, 250, 500, 750, 100]
    tracks = [2, 3, 6, 4, 9, 1]
    sampler = ScenePaddedBatchSampler(
        frames, tracks, frames_per_batch=cap, max_batch_size=8, shuffle=False
    )
    assert sorted(i for b in sampler for i in b) == list(range(len(frames)))


def test_quantisation_collapses_shapes_and_keeps_the_bound():
    frames = list(range(601, 649))          # 48 distinct lengths, all -> 650
    tracks = [1] * len(frames)
    cap = 2600

    def shapes(sampler):
        return {
            (sum(tracks[i] for i in b), sampler.width(max(frames[i] for i in b)))
            for b in sampler
        }

    plain = ScenePaddedBatchSampler(frames, tracks, frames_per_batch=cap,
                                    max_batch_size=32, shuffle=False)
    quant = ScenePaddedBatchSampler(frames, tracks, frames_per_batch=cap,
                                    max_batch_size=32, shuffle=False, frame_quantum=50)

    assert len(shapes(plain)) > 5
    assert len(shapes(quant)) == 1
    for b in quant:
        width = quant.width(max(frames[i] for i in b))
        assert width * sum(tracks[i] for i in b) <= cap
    assert sorted(i for b in quant for i in b) == list(range(len(frames)))


def test_quantisation_never_widens_past_the_longest_clip():
    s = ScenePaddedBatchSampler([750] * 8, [2] * 8, frames_per_batch=6000,
                                max_batch_size=32, shuffle=False, frame_quantum=400)
    assert s.width(750) == 750


def test_quantum_zero_is_the_identity():
    s = ScenePaddedBatchSampler([613, 431], [1, 1], frames_per_batch=6000,
                                max_batch_size=8, shuffle=False, frame_quantum=0)
    assert s.width(613) == 613 and s.width(431) == 431
