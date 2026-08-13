"""Tests for rnd_convergence.mars_rover.

The anchor rung has to be trustworthy before anything measured against it means much, so
the dynamics are asserted directly rather than through the agent.
"""

import numpy as np
import pytest

from rnd_convergence.mars_rover import MarsRover


class TestSpec:
    def test_starts_in_the_middle_cell(self):
        obs, _ = MarsRover().reset(seed=0)
        assert obs == 2

    def test_spaces_match_the_one_dimensional_rung(self):
        env = MarsRover()
        assert env.observation_space.n == 5
        assert env.action_space.n == 2

    def test_actions_move_one_cell_each_way(self):
        env = MarsRover()
        env.reset(seed=0)
        assert env.step(1)[0] == 3
        assert env.step(0)[0] == 2

    def test_movement_is_clamped_at_both_ends(self):
        env = MarsRover(horizon=100)
        env.reset(seed=0)
        for _ in range(10):
            position = env.step(1)[0]
        assert position == 4

        env.reset(seed=0)
        for _ in range(10):
            position = env.step(0)[0]
        assert position == 0

    def test_reward_is_that_of_the_cell_landed_in(self):
        env = MarsRover()
        env.reset(seed=0)
        assert env.step(1)[1] == 0.0  # cell 3
        assert env.step(1)[1] == 10.0  # cell 4
        env.reset(seed=0)
        assert env.step(0)[1] == 0.0  # cell 1
        assert env.step(0)[1] == 1.0  # cell 0

    def test_truncates_at_the_horizon_and_never_terminates(self):
        env = MarsRover(horizon=4)
        env.reset(seed=0)
        flags = [env.step(1)[2:4] for _ in range(4)]
        assert [t for t, _ in flags] == [False] * 4
        assert [tr for _, tr in flags] == [False, False, False, True]

    def test_optimal_return_is_reachable(self):
        # Two steps right, then park on the 10-reward cell for the rest of the horizon.
        env = MarsRover()
        env.reset(seed=0)
        total = sum(env.step(1)[1] for _ in range(10))
        assert total == 90.0

    def test_rejects_invalid_actions(self):
        env = MarsRover()
        env.reset(seed=0)
        with pytest.raises(ValueError, match="invalid action"):
            env.step(2)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"rewards": (1.0,)}, "at least 2 cells"),
            ({"transition_probability": 1.5}, "transition_probability"),
            ({"horizon": 0}, "horizon"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MarsRover(**kwargs)


class TestStochasticity:
    def test_deterministic_by_default(self):
        env = MarsRover(horizon=50)
        env.reset(seed=0)
        assert all(env.step(1)[0] == min(3 + i, 4) for i in range(5))

    def test_flipping_actually_flips_sometimes(self):
        env = MarsRover(horizon=1_000, transition_probability=0.5)
        env.reset(seed=0)
        positions = [env.step(1)[0] for _ in range(200)]
        # A always-right rover would pin to cell 4; flips must pull it back down.
        assert min(positions) < 4

    def test_flipping_is_reproducible_from_the_reset_seed(self):
        def roll():
            env = MarsRover(horizon=100, transition_probability=0.5)
            env.reset(seed=7)
            return [env.step(1)[0] for _ in range(50)]

        assert roll() == roll()

    def test_never_flips_at_probability_one(self):
        env = MarsRover(horizon=100, transition_probability=1.0)
        env.reset(seed=3)
        assert np.all(
            np.array([env.step(0)[0] for _ in range(10)]) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
