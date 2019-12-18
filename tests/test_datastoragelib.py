# -*- coding: utf-8 -*-
import pytest
from datastoragelib import DataStorage


class TestDatastoragelib:

    def test_new_datastorage(self):
        ds = DataStorage(size=100, frame_height=8, frame_width=4, agent_history_length=5, batch_size=32)
        assert (ds.count == 0)
        assert (ds.current == 0)
        assert (ds.frame_height == 8)
        assert (ds.frame_width == 4)
        assert (len(ds.actions) == 100)
        assert (len(ds.rewards) == 100)
        assert (len(ds.terminal_flags) == 100)
        assert (len(ds.frames) == 100)
        assert (len(ds.frames[0]) == 8)
        assert (len(ds.frames[0][0]) == 4)
        assert (len(ds.states) == 32)
        assert (len(ds.states[0]) == 5)
        assert (len(ds.states[0][0]) == 8)
        assert (len(ds.states[0][0][0]) == 4)
        assert (len(ds.new_states) == 32)
        assert (len(ds.new_states[0]) == 5)
        assert (len(ds.new_states[0][0]) == 8)
        assert (len(ds.new_states[0][0][0]) == 4)

    def test_add_experience(self):
        ds = DataStorage(size=10, frame_height=5, frame_width=2)
        list = [[255 for x in range(2)] for y in range(5)]
        ds.add_experience(1, list, 0, True)
        assert (ds.count == 1)
        assert (ds.current == 1)
        assert (ds.frames[ds.current - 1] == list)
        list = [[255 for x in range(3)] for y in range(5)]
        try:
            ds.add_experience(1, list, 0, True)
            assert False
        except ValueError:
            assert True
        list = [[256 for x in range(3)] for y in range(5)]
        try:
            ds.add_experience(1, list, 0, True)
            assert False
        except OverflowError:
            assert True

    def test_get_state(self):
        ds = DataStorage(size=10, frame_height=5, frame_width=2, agent_history_length=4)
        list = [[255 for x in range(2)] for y in range(5)]
        for i in range(10):
            ds.add_experience(1, list, 1, True)
        state = ds._get_state(4)
        assert (len(state) == 4)

    def test_get_valid_indices(self):
        ds = DataStorage(size=1000, frame_height=5, frame_width=2, agent_history_length=4)
        list = [[255 for x in range(2)] for y in range(5)]
        for i in range(100):
            ds.add_experience(1, list, 1, False)
        assert (len(ds.indices) == 4)
        assert (ds.indices[0] == 0)
        ds._get_valid_indices()
        assert (ds.indices[0] != 0)

    def test_get_minibatch(self):
        BS = 32
        ds = DataStorage(size=1000000, frame_height=84, frame_width=84, agent_history_length=4, batch_size=BS)
        list = [[127 for x in range(120)] for y in range(120)]
        for i in range(900):
            ds.add_experience(1, list, 1, False)
        s, a, r, s_, t = ds.get_minibatch()
        assert (len(s) == len(a) == len(r) == len(s_) == len(t) == BS)
