"""Scene mixers: no lookahead, no cross-scene leakage, identity at init."""

import pytest
import torch

from cast_asd.models.modules.scene_mixers import build_scene_mixer

CFG = {"d_model": 32, "n_heads": 4, "m_ff": 2, "n_layers": 2, "dropout": 0.0}

# LoCoNet reshuffles its roster on every pass; pin it so two passes differ only
# by the perturbation. The roster is built from the track mask alone.
MIXERS = {
    "cast": {"mechanism": "cast"},
    "cast_time_speaker": {"mechanism": "cast", "order": "time_speaker"},
    "loconet": {"mechanism": "loconet", "roster_shuffle": False},
}


def _grouping(tracks_per_scene):
    index, slot = [], []
    for scene, count in enumerate(tracks_per_scene):
        index += [scene] * count
        slot += list(range(count))
    mask = torch.zeros(len(tracks_per_scene), max(tracks_per_scene), dtype=torch.bool)
    for scene, count in enumerate(tracks_per_scene):
        mask[scene, :count] = True
    return torch.tensor(index), torch.tensor(slot), mask


def _mixer(name, open_gate=True, **overrides):
    torch.manual_seed(0)
    block = build_scene_mixer({**CFG, **MIXERS[name], **overrides}, CFG["d_model"]).eval()
    if open_gate:
        with torch.no_grad():
            block.gate.fill_(1.0)          # a closed gate would pass everything
    return block


@pytest.mark.parametrize("name", MIXERS)
def test_no_mixer_reads_a_later_frame(name):
    index, slot, mask = _grouping([3, 2])
    block = _mixer(name)
    x = torch.randn(5, 9, CFG["d_model"])

    with torch.no_grad():
        reference = block(x, index, slot, mask)
        for t in range(1, 9):
            perturbed = x.clone()
            perturbed[:, t:] += 7.0
            output = block(perturbed, index, slot, mask)
            assert torch.allclose(output[:, :t], reference[:, :t], atol=1e-5), (
                f"{name}: perturbing frames >= {t} changed an earlier frame"
            )


@pytest.mark.parametrize("name", MIXERS)
def test_non_causal_mixer_does_read_ahead(name):
    index, slot, mask = _grouping([3])
    block = _mixer(name, causal=False)
    x = torch.randn(3, 9, CFG["d_model"])
    with torch.no_grad():
        reference = block(x, index, slot, mask)
        perturbed = x.clone()
        perturbed[:, 5:] += 7.0
        output = block(perturbed, index, slot, mask)
    assert not torch.allclose(output[:, :5], reference[:, :5], atol=1e-4)


@pytest.mark.parametrize("name", MIXERS)
def test_every_mixer_sees_its_neighbours(name):
    """Moving only the neighbour at t=3 changes face 0 at t=3, and nothing before."""
    index, slot, mask = _grouping([2])
    block = _mixer(name)
    x = torch.randn(2, 6, CFG["d_model"])

    with torch.no_grad():
        reference = block(x, index, slot, mask)
        perturbed = x.clone()
        perturbed[1, 3] += 7.0
        output = block(perturbed, index, slot, mask)

    assert not torch.allclose(output[0, 3], reference[0, 3], atol=1e-4)
    assert torch.allclose(output[0, :3], reference[0, :3], atol=1e-5)


@pytest.mark.parametrize("name", MIXERS)
def test_no_mixer_leaks_between_scenes(name):
    index, slot, mask = _grouping([2, 2])
    block = _mixer(name)
    x = torch.randn(4, 5, CFG["d_model"])

    with torch.no_grad():
        reference = block(x, index, slot, mask)
        perturbed = x.clone()
        perturbed[2:] += 7.0                        # the whole second scene
        output = block(perturbed, index, slot, mask)

    assert torch.allclose(output[:2], reference[:2], atol=1e-6)


@pytest.mark.parametrize("name", MIXERS)
def test_padded_track_slots_are_invisible(name):
    """A 2-face scene must score the same when the batch is 3 slots wide."""
    block = _mixer(name)
    x = torch.randn(2, 5, CFG["d_model"])
    index, slot, mask = _grouping([2])
    with torch.no_grad():
        tight = block(x, index, slot, mask)
        padded = block(x, index, slot, torch.tensor([[True, True, False]]))
    assert torch.allclose(tight, padded, atol=1e-6)


@pytest.mark.parametrize("name", ["cast", "cast_time_speaker"])
def test_cast_is_permutation_equivariant(name):
    index, slot, mask = _grouping([3])
    block = _mixer(name)
    x = torch.randn(3, 5, CFG["d_model"])
    with torch.no_grad():
        straight = block(x, index, slot, mask)
        swapped = block(x, index, torch.tensor([2, 1, 0]), mask)
    assert torch.allclose(straight, swapped, atol=1e-5)


@pytest.mark.parametrize("name", MIXERS)
def test_a_closed_gate_is_exactly_the_identity(name):
    index, slot, mask = _grouping([3, 2])
    block = _mixer(name, open_gate=False)            # gate left at its zero init
    x = torch.randn(5, 5, CFG["d_model"])
    with torch.no_grad():
        assert torch.equal(block(x, index, slot, mask), x)


def test_factory():
    assert build_scene_mixer({**CFG}, CFG["d_model"]) is None
    assert build_scene_mixer({**CFG, "mechanism": "none"}, CFG["d_model"]) is None
    with pytest.raises(ValueError, match="unknown asd.scene.mechanism"):
        build_scene_mixer({**CFG, "mechanism": "telepathy"}, CFG["d_model"])
    with pytest.raises(ValueError, match="order"):
        build_scene_mixer({**CFG, "mechanism": "cast", "order": "sideways"}, CFG["d_model"])


def test_width_mismatch_is_an_error():
    index, slot, mask = _grouping([2])
    with pytest.raises(ValueError, match="expected width"):
        _mixer("cast")(torch.randn(2, 4, CFG["d_model"] + 1), index, slot, mask)
