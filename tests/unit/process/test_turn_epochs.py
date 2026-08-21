"""The pooled `turn` epochs built by the turn annotation.

`turn` is a declared epoch type (see `DatasetConfig.h5_kdic`) and the chunk
the per-turn parameters are tracked in, but `turn_annotation` used to build only
the left and the right epochs. Every analysis reaching for a per-turn quantity -
the `patch` graphgroup of the food-patch assays among them - therefore died with
`KeyError: 'turn'`.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def turn_chunks(real_dataset):
    """The per-agent chunk dictionaries produced by the turn annotation."""
    return real_dataset.turn_annotation()


@pytest.mark.fast
class TestPooledTurnEpochs:
    def test_turn_is_a_declared_epoch_type(self, real_dataset):
        """The storage schema has always expected the pooled epochs to exist."""
        from larvaworld.lib.util import nam

        schema = {p for v in real_dataset.config.h5_kdic.values() for p in v}
        assert set(nam.chunk_track("turn", ["dur", "start", "stop"])) <= schema

    def test_every_agent_gets_pooled_turn_epochs(self, turn_chunks):
        assert turn_chunks
        for id, D in turn_chunks.items():
            assert "turn" in D, f"{id} has no pooled turn epochs"

    def test_pooled_epochs_are_the_left_and_the_right_ones(self, turn_chunks):
        for D in turn_chunks.values():
            assert D.turn.ndim == 2 and D.turn.shape[1] == 2
            assert D.turn.shape[0] == D.Lturn.shape[0] + D.Rturn.shape[0]

    def test_left_turns_come_first(self, turn_chunks):
        """`turn_mode_annotation` indexes `turn_amp` by left-then-right position,
        so the pooled epochs have to be built in that same order."""
        for D in turn_chunks.values():
            nL = D.Lturn.shape[0]
            np.testing.assert_array_equal(D.turn[:nL], D.Lturn)
            np.testing.assert_array_equal(D.turn[nL:], D.Rturn)
            np.testing.assert_allclose(D.turn_amp[:nL], D.Lturn_amp)
            np.testing.assert_allclose(D.turn_dur[:nL], D.Lturn_dur)

    def test_derived_arrays_stay_aligned_with_the_epochs(self, turn_chunks):
        for D in turn_chunks.values():
            n = D.turn.shape[0]
            assert D.turn_dur.shape[0] == n
            assert D.turn_amp.shape[0] == n
            assert D.turn_vel_max.shape[0] == n

    def test_epochs_are_start_stop_pairs(self, turn_chunks):
        for D in turn_chunks.values():
            if D.turn.shape[0] == 0:
                continue
            assert np.all(D.turn[:, 1] >= D.turn[:, 0])
