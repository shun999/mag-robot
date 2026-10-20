import json
import re
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from dino_ingot.ur3 import (
    DecisionState,
    RobotChannel,
    UR3Config,
    WristController,
    load_ur3_config,
)


def test_example_and_config_are_independent_of_model(tmp_path):
    config = load_ur3_config(Path(__file__).parents[1] / "ur3.example.json")
    assert config == UR3Config()
    path = tmp_path / "ur3.json"
    path.write_text(json.dumps({"speed": 0.2}), encoding="utf-8")
    assert load_ur3_config(path).speed == 0.2


@pytest.mark.parametrize("changes", [
    {"speed": float("nan")}, {"acceleration": 0}, {"speed": True},
    {"stop_deceleration": float("inf")}, {"stable_count": 0}, {"stable_count": True},
    {"robot_port": 65536}, {"listen_port": False}, {"normal_direction": True},
    {"anomaly_direction": 1}, {"normal_direction": 0}, {"listen_host": "0.0.0.0"},
    {"robot_host": '192.168.11.20")\nstopj(2)'}, {"heartbeat_interval": 0.3},
    {"robot_host": 3232238356},
    {"motion_period": 0.5}, {"inference_timeout": 0.1}, {"shutdown_timeout": 0.5},
])
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        replace(UR3Config(), **changes).validate()


def test_decisions_require_consecutive_results_and_stop_before_reversal():
    state = DecisionState(UR3Config(), 0)
    for i, expected in enumerate((0, 0, 1, 1), start=1):
        state.update("NORMAL", i * 0.1, i * 0.1)
        assert state.direction == expected
    state.update("ANOMALY", 0.5, 0.5)
    assert (state.direction, state.count) == (0, 1)
    state.update("NORMAL", 0.6, 0.6)
    assert (state.direction, state.count) == (0, 1)
    for i, expected in enumerate((0, 0, -1), start=7):
        state.update("ANOMALY", i * 0.1, i * 0.1)
        assert state.direction == expected


@pytest.mark.parametrize("verdict", ["NO OBJECT", "INVALID"])
def test_invalid_region_stops_and_requires_new_confirmation(verdict):
    state = DecisionState(UR3Config(), 0)
    for i in range(1, 4):
        state.update("NORMAL", i * 0.1, i * 0.1)
    state.update(verdict, 0.4, 0.4)
    assert (state.direction, state.count, state.candidate) == (0, 0, None)
    state.update("NORMAL", 0.5, 0.5)
    assert (state.direction, state.count) == (0, 1)


def test_direction_mapping_and_immediate_mode():
    config = replace(UR3Config(), stable_count=1, normal_direction=-1, anomaly_direction=1)
    state = DecisionState(config, 0)
    state.update("NORMAL", 0.1, 0.1)
    assert state.direction == -1
    state.update("ANOMALY", 0.2, 0.2)
    assert state.direction == 1


def test_freshness_uses_frame_start_and_cannot_be_renewed_after_expiry():
    state = DecisionState(replace(UR3Config(), stable_count=1), 0)
    state.update("NORMAL", 0.1, 1.0)
    state.check_fresh(2.09)
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        state.update("NORMAL", 2.1, 2.1)
    assert state.direction == 0
    # Even an immediately completed new result cannot resume an expired session.
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        state.update("NORMAL", 2.2, 2.2)


@pytest.mark.parametrize("started", [-1, 0.2, float("nan"), float("inf")])
def test_bad_frame_timestamps_rejected(started):
    state = DecisionState(UR3Config(), 0)
    with pytest.raises(ValueError):
        state.update("NORMAL", started, 0.1)


class Wire:
    """A socket with partial reads and an independently scripted robot reply."""

    def __init__(self, replies=(), *, bad_ack=False, timeout=False):
        self.replies = deque(replies)
        self.sent = []
        self.closed = False
        self.bad_ack, self.timeout = bad_ack, timeout

    def settimeout(self, seconds):
        assert seconds > 0

    def sendall(self, data):
        self.sent.append(data)
        sequence, direction = (int(v) for v in data.decode().strip()[1:-1].split(","))
        if direction == 9:
            self.replies.extend(bytes([v]) for v in b"STOPPED\n")
        else:
            self.replies.extend(bytes([v]) for v in struct.pack("!i", sequence + self.bad_ack))

    def recv(self, size):
        if self.timeout:
            raise socket.timeout("missing ACK")
        return self.replies.popleft() if self.replies else b""

    def close(self):
        self.closed = True


def test_fragmented_handshake_ack_and_stop_confirmation():
    wire = Wire(bytes([v]) for v in b"READY ingot-v1\n")
    channel = RobotChannel(wire, UR3Config())
    channel.ready()
    channel.command(1)
    channel.command(-1)
    channel.command(0)
    channel.finish()
    assert wire.sent == [b"(1,1)\n", b"(2,-1)\n", b"(3,0)\n", b"(4,9)\n"]


@pytest.mark.parametrize("reply", [b"wrong\n", b"READY", b"x" * 128])
def test_bad_ready_or_disconnection_rejected(reply):
    channel = RobotChannel(Wire(bytes([v]) for v in reply), UR3Config())
    with pytest.raises((RuntimeError, ConnectionError)):
        channel.ready()


@pytest.mark.parametrize("wire,error", [
    (Wire(bad_ack=True), RuntimeError), (Wire(timeout=True), socket.timeout),
])
def test_ack_must_arrive_and_match_sequence(wire, error):
    with pytest.raises(error):
        RobotChannel(wire, UR3Config()).command(1)


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("controller did not reach expected state")
        threading.Event().wait(0.005)


def test_dry_run_never_opens_socket_and_stops_while_inference_is_blocked(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run opened a socket")

    monkeypatch.setattr("dino_ingot.ur3.socket.socket", forbidden)
    now = [0.0]
    controller = WristController(UR3Config(), dry_run=True, clock=lambda: now[0])
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        with controller:
            for i in range(1, 4):
                now[0] = i * 0.1
                controller.publish("NORMAL", now[0])
            assert "cmd=+1" in controller.status()
            # No publish/check call from the inference thread: worker must stop itself.
            now[0] = 2.31
            wait_until(lambda: controller.stopped_confirmed)
            assert "cmd=+0" in controller.status()
    assert controller._worker.is_alive() is False


def attach_wire(monkeypatch, controller, wire):
    def open_channel():
        controller._channel = RobotChannel(wire, controller.config)

    monkeypatch.setattr(controller, "_open", open_channel)


def test_normal_exit_requests_stop_and_waits_for_confirmation(monkeypatch):
    controller = WristController(UR3Config())
    wire = Wire()
    attach_wire(monkeypatch, controller, wire)
    with controller:
        wait_until(lambda: len(wire.sent) >= 1)
    assert wire.sent[0] == b"(1,0)\n"
    assert wire.sent[-1].endswith(b",9)\n")
    assert controller.stopped_confirmed and wire.closed
    assert not controller._worker.is_alive()


def test_inference_timeout_sends_shutdown_without_gui_cooperation(monkeypatch):
    now = [0.0]
    controller = WristController(UR3Config(), clock=lambda: now[0])
    wire = Wire()
    attach_wire(monkeypatch, controller, wire)
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        with controller:
            for i in range(1, 4):
                now[0] = i * 0.1
                controller.publish("NORMAL", now[0])
            wait_until(lambda: any(b",1)" in command for command in wire.sent))
            now[0] = 2.31
            wait_until(lambda: wire.closed)
    assert wire.sent[-1].endswith(b",9)\n")
    assert controller.stopped_confirmed


def test_brief_stop_is_not_overwritten_by_new_confirmed_result(monkeypatch):
    commands = []
    rotating, release_ack = threading.Event(), threading.Event()

    class GatedChannel:
        first_rotation = True

        def command(self, direction):
            commands.append(direction)
            if direction == 1 and self.first_rotation:
                self.first_rotation = False
                rotating.set()
                assert release_ack.wait(2)

        def finish(self):
            commands.append(9)

        def close(self):
            pass

    controller = WristController(replace(UR3Config(), stable_count=1))

    def open_channel():
        controller._channel = GatedChannel()

    monkeypatch.setattr(controller, "_open", open_channel)
    try:
        with controller:
            controller.publish("NORMAL", time.monotonic())
            assert rotating.wait(2)
            # Both updates arrive while the worker is waiting for the rotation ACK.
            controller.publish("NO OBJECT", time.monotonic())
            controller.publish("NORMAL", time.monotonic())
            release_ack.set()
            wait_until(lambda: len(commands) >= 4)
            assert commands[:4] == [0, 1, 0, 1]
    finally:
        release_ack.set()


@pytest.mark.parametrize("exception", [ValueError("camera failed"), KeyboardInterrupt()])
def test_pipeline_errors_and_interrupt_stop_before_return(monkeypatch, exception):
    controller = WristController(UR3Config())
    wire = Wire()
    attach_wire(monkeypatch, controller, wire)
    with pytest.raises(type(exception)):
        with controller:
            wait_until(lambda: bool(wire.sent))
            raise exception
    assert controller.stopped_confirmed and wire.closed


def test_protocol_error_closes_stream_without_claiming_stop(monkeypatch, capsys):
    controller = WristController(UR3Config())
    wire = Wire(bad_ack=True)
    attach_wire(monkeypatch, controller, wire)
    with pytest.raises(RuntimeError, match="ACK"):
        with controller:
            wait_until(lambda: wire.closed)
    assert not controller.stopped_confirmed
    assert wire.sent == [b"(1,0)\n"]
    assert "停止未確認" in capsys.readouterr().out


def test_shutdown_timeout_is_reported(monkeypatch, capsys):
    controller = WristController(UR3Config())
    wire = Wire()
    attach_wire(monkeypatch, controller, wire)
    with pytest.raises(RuntimeError, match="停止未確認"):
        with controller:
            wait_until(lambda: bool(wire.sent))
            wire.timeout = True
    assert not controller.stopped_confirmed and wire.closed
    assert "停止未確認" in capsys.readouterr().out


def test_failed_startup_releases_listener_and_primary(monkeypatch):
    controller = WristController(UR3Config())
    wire = Wire()

    def fail_open():
        controller._primary = wire
        controller._listener = wire
        raise ConnectionRefusedError("robot unavailable")

    monkeypatch.setattr(controller, "_open", fail_open)
    with pytest.raises(ConnectionRefusedError):
        with controller:
            pytest.fail("startup should have failed")
    assert wire.closed


def test_unknown_config_fields_rejected(tmp_path):
    path = tmp_path / "ur3.json"
    path.write_text(json.dumps({**asdict(UR3Config()), "speeed": 2}), encoding="utf-8")
    with pytest.raises(TypeError):
        load_ur3_config(path)


def test_upload_reverse_connection_and_heartbeat_over_loopback():
    """Exercise real TCP framing and startup/cleanup against a local robot peer.

    This peer implements the wire protocol, not the URScript motion runtime.
    """
    errors, commands, uploads = [], [], []
    with socket.socket() as server, socket.socket() as port_probe:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(3)
        port_probe.bind(("127.0.0.1", 0))
        callback_port = port_probe.getsockname()[1]
        port_probe.close()
        config = replace(
            UR3Config(), robot_host="127.0.0.1", robot_port=server.getsockname()[1],
            listen_host="127.0.0.1", listen_port=callback_port,
        )

        def robot_peer():
            try:
                primary, _ = server.accept()
                with primary:
                    primary.settimeout(3)
                    script = b""
                    while not script.endswith(b"\nend\n"):
                        chunk = primary.recv(4096)
                        if not chunk:
                            raise AssertionError("incomplete upload")
                        script += chunk
                    uploads.append(script.decode("ascii"))
                    address = re.search(r'socket_open\("([0-9.]+)", (\d+),', uploads[0])
                    with socket.create_connection((address[1], int(address[2])), timeout=3) as peer:
                        peer.sendall(b"READY ingot-v1\n")
                        # The upload socket publishes unrelated state alongside the callback.
                        primary.sendall(b"state" * 100)
                        with peer.makefile("rb") as stream:
                            while True:
                                line = stream.readline()
                                if not line:
                                    raise AssertionError("shutdown command missing")
                                seq, direction = (int(v) for v in line.strip()[1:-1].split(b","))
                                commands.append(direction)
                                if direction == 9:
                                    peer.sendall(b"STOPPED\n")
                                    break
                                peer.sendall(struct.pack("!i", seq))
                    # Keep the upload connection alive until the controller cleans it up.
                    while primary.recv(4096):
                        pass
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=robot_peer, daemon=True)
        thread.start()
        try:
            with WristController(config) as controller:
                for _ in range(3):
                    controller.publish("NORMAL", time.monotonic())
                wait_until(lambda: 1 in commands)
                controller.publish("INVALID", time.monotonic())
                wait_until(lambda: commands[-1] == 0)
            assert controller.stopped_confirmed
        finally:
            thread.join(4)
        assert not thread.is_alive()
        assert not errors
        assert len(uploads) == 1 and commands[-1] == 9
        assert not controller._worker.is_alive() and not controller._drainer.is_alive()
