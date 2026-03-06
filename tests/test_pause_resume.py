"""Tests for true pause/resume in BaseTrainer."""
import time
import threading
import pytest


class FakeTrainer:
    """Minimal concrete trainer for testing pause logic."""

    def __init__(self):
        self._training_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self.steps_taken = 0

    @property
    def training_paused(self):
        return self._training_paused

    @training_paused.setter
    def training_paused(self, value):
        self._training_paused = value
        if value:
            self._pause_event.clear()
        else:
            self._pause_event.set()

    def check_pause(self, timeout=5.0):
        if not self._training_paused:
            return True
        return self._pause_event.wait(timeout=timeout)

    def fake_training_loop(self, max_steps=100):
        for _ in range(max_steps):
            if not self.check_pause(timeout=0.5):
                break
            self.steps_taken += 1
            time.sleep(0.01)


def test_not_paused_by_default():
    t = FakeTrainer()
    assert not t.training_paused
    assert t.check_pause() is True


def test_pause_blocks_training():
    t = FakeTrainer()
    t.training_paused = True
    start = time.time()
    result = t.check_pause(timeout=0.2)
    elapsed = time.time() - start
    assert result is False
    assert elapsed >= 0.15


def test_resume_unblocks_training():
    t = FakeTrainer()
    t.training_paused = True

    def resume_after_delay():
        time.sleep(0.1)
        t.training_paused = False

    thread = threading.Thread(target=resume_after_delay)
    thread.start()
    result = t.check_pause(timeout=2.0)
    assert result is True
    thread.join()


def test_pause_stops_step_accumulation():
    t = FakeTrainer()
    train_thread = threading.Thread(target=t.fake_training_loop, args=(1000,))
    train_thread.start()
    time.sleep(0.1)
    steps_before_pause = t.steps_taken
    t.training_paused = True
    time.sleep(0.2)
    steps_while_paused = t.steps_taken
    assert steps_while_paused - steps_before_pause <= 1
    t.training_paused = False
    time.sleep(0.1)
    assert t.steps_taken > steps_while_paused
    train_thread.join(timeout=5)
