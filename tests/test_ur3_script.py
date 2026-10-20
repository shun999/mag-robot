"""Regressions for the generated CB3 receiver and watchdog.

The small adapter below executes the generated thread bodies with deterministic
read/sync suspension points. It is not a URScript interpreter: actual CB3 typing,
scheduling, and motion still need URSim or hardware validation.
"""

import ast
import re
from dataclasses import replace

import pytest

from dino_ingot.ur3 import UR3Config, control_script


class ScriptThreads:
    def __init__(self, config=UR3Config()):
        self.script = control_script(config)
        self.acks = []
        self.ack_ok = True
        self.state = {
            name: ast.literal_eval(value)
            for name, value in re.findall(r"^  global (\w+) = (.+)$", self.script, re.M)
        }
        shared = ", ".join(self.state)

        # Deliberately match 3.5.1's maximum of two arguments. Passing timeout
        # would fail here as it did on the user's teach pendant.
        def read(number, socket_name="socket_0"):
            assert number == 2 and socket_name == "ingot"
            packet = yield "read"
            return packet

        def ack(sequence, socket_name):
            assert socket_name == "ingot"
            self.acks.append(sequence)
            return self.ack_ok

        self.state.update(socket_read_ascii_float=read, socket_send_int=ack)
        for name in ("ingot_receive", "ingot_watchdog"):
            body = re.search(
                rf"^  thread {name}\(\):\n(.*?)^  end$", self.script, re.M | re.S
            )[1]
            lines = [f"def {name}():", f"    global {shared}"]
            for line in body.splitlines():
                if line.strip() in ("end", "enter_critical", "exit_critical"):
                    continue
                line = re.sub(r"^(\s*)local ", r"\1", line)
                line = line.replace("sync()", '(yield "sync")')
                line = line.replace("= socket_read_ascii_float(", "= yield from read(")
                lines.append(line)
            self.state["read"] = read
            exec(compile("\n".join(lines), f"<CB3 {name} logic>", "exec"), self.state)
        self.receiver = self.state["ingot_receive"]()
        self.watchdog = self.state["ingot_watchdog"]()
        assert next(self.receiver) == "read"
        assert next(self.watchdog) == "sync"

    def receive(self, packet):
        try:
            self.receiver.send(packet)
        except StopIteration:
            pass

    def cycles(self, count):
        for _ in range(count):
            try:
                next(self.watchdog)
            except StopIteration:
                break


def test_socket_read_uses_cb3_351_signature():
    script = control_script(UR3Config())
    calls = re.findall(r"socket_read_ascii_float\([^\n]+\)", script)
    assert calls
    for call in calls:
        expression = ast.parse(call, mode="eval").body
        assert len(expression.args) == 2
        assert expression.keywords == []


@pytest.mark.parametrize("timeout,ticks", [(0.5, 63), (0.501, 63), (0.504, 63), (1.0, 125)])
def test_watchdog_expires_while_receiver_is_still_blocked(timeout, ticks):
    robot = ScriptThreads(replace(UR3Config(), communication_timeout=timeout))
    robot.receive([2, 1, 1])
    robot.cycles(ticks - 1)
    assert robot.state["ingot_alive"] and robot.state["ingot_direction"] == 1
    robot.cycles(1)
    assert not robot.state["ingot_alive"]
    assert robot.state["ingot_direction"] == 0
    assert robot.acks == [1]


def test_only_valid_heartbeats_rearm_and_late_packets_cannot_restart():
    robot = ScriptThreads()
    robot.receive([2, 1, 1])
    robot.cycles(40)
    robot.receive([2, 2, -1])
    assert robot.state["ingot_lease_ticks"] == 0
    robot.cycles(62)
    assert robot.state["ingot_alive"]
    robot.cycles(1)
    robot.receive([2, 3, 1])
    assert not robot.state["ingot_alive"] and robot.state["ingot_direction"] == 0
    assert robot.acks == [1, 2]


@pytest.mark.parametrize("packet", [[0, float("nan"), float("nan")], [2, 99, 1], [2, 2, 7]])
def test_invalid_or_failed_read_stops_without_renewing_lease(packet):
    robot = ScriptThreads()
    robot.receive([2, 1, 1])
    robot.cycles(20)
    robot.receive(packet)
    assert not robot.state["ingot_alive"] and robot.state["ingot_direction"] == 0
    assert robot.state["ingot_lease_ticks"] == 20
    assert robot.acks == [1]


def test_zero_heartbeats_rearm_and_preserve_stop_request():
    robot = ScriptThreads()
    robot.receive([2, 1, 0])
    robot.cycles(40)
    robot.receive([2, 2, 0])
    assert robot.state["ingot_lease_ticks"] == 0
    assert robot.state["ingot_stop_requested"]
    robot.receive([2, 3, 1])
    assert robot.state["ingot_stop_requested"]  # Motion thread must consume this stop.
    assert robot.acks == [1, 2, 3]


def test_shutdown_stops_without_waiting_for_read_timeout():
    robot = ScriptThreads()
    robot.receive([2, 1, 1])
    robot.receive([2, 2, 9])
    assert not robot.state["ingot_alive"] and robot.state["ingot_direction"] == 0
    assert robot.acks == [1]  # Shutdown is acknowledged by STOPPED in the motion thread.
    robot.cycles(1)
    assert robot.watchdog.gi_frame is None


def test_failed_ack_stops_receiver():
    robot = ScriptThreads()
    robot.ack_ok = False
    robot.receive([2, 1, 1])
    assert not robot.state["ingot_alive"] and robot.state["ingot_direction"] == 0


def test_threads_start_before_ready_and_are_killed_before_stop_confirmation():
    script = control_script(UR3Config())
    ready = script.index('socket_send_line("READY ingot-v1"')
    assert script.index("run ingot_receive()") < ready
    assert script.index("run ingot_watchdog()") < ready
    stopped = script.index('socket_send_line("STOPPED"')
    stop_motion = script.rindex("stopj(")
    assert stop_motion < script.index("kill receiver") < stopped
    assert stop_motion < script.index("kill watchdog") < stopped
