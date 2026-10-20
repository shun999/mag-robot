"""Ingot decisions and a leased, reverse-socket UR3 wrist controller.

Only the URScript main thread issues motion commands. Separate robot threads
receive heartbeats and enforce their deadline, independently of blocking reads.
"""

import ipaddress
import json
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

CB3_CYCLE_SECONDS = 0.008  # CB3 sync() consumes the remainder of a 125 Hz frame.


@dataclass
class UR3Config:
    robot_host: str = "192.168.11.20"
    robot_port: int = 30002
    listen_host: str = "192.168.11.5"
    listen_port: int = 50001
    speed: float = 1.0
    acceleration: float = 0.5
    stop_deceleration: float = 2.0
    normal_direction: int = 1
    anomaly_direction: int = -1
    stable_count: int = 3
    heartbeat_interval: float = 0.1
    communication_timeout: float = 0.5
    inference_timeout: float = 2.0
    motion_period: float = 0.05
    connect_timeout: float = 5.0
    shutdown_timeout: float = 2.0

    def validate(self) -> None:
        for name in ("robot_host", "listen_host"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} はIPv4アドレスの文字列です")
            address = ipaddress.IPv4Address(getattr(self, name))
            if address.is_unspecified or address.is_multicast:
                raise ValueError(f"{name} は接続可能なIPv4アドレスを指定してください")
        for name in ("robot_port", "listen_port"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError(f"{name} は1〜65535の整数です")
        if type(self.stable_count) is not int or self.stable_count < 1:
            raise ValueError("stable_count は正の整数です")
        for name in ("normal_direction", "anomaly_direction"):
            if type(getattr(self, name)) is not int or getattr(self, name) not in (-1, 1):
                raise ValueError(f"{name} は1または-1です")
        if self.normal_direction == self.anomaly_direction:
            raise ValueError("正常と異常には逆の方向を指定してください")
        for name in (
            "speed", "acceleration", "stop_deceleration", "heartbeat_interval",
            "communication_timeout", "inference_timeout", "motion_period",
            "connect_timeout", "shutdown_timeout",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} は有限の正の数です")
        if self.heartbeat_interval * 2 > self.communication_timeout:
            raise ValueError("communication_timeout は heartbeat_interval の2倍以上が必要です")
        if self.motion_period >= self.communication_timeout:
            raise ValueError("motion_period は communication_timeout より短くしてください")
        if self.inference_timeout <= self.heartbeat_interval:
            raise ValueError("inference_timeout は heartbeat_interval より長くしてください")
        stopping_time = self.speed / self.stop_deceleration + self.motion_period
        if self.shutdown_timeout <= stopping_time + self.communication_timeout:
            raise ValueError("shutdown_timeout は減速時間・運動周期・通信時間より長くしてください")


def load_ur3_config(path: Path) -> UR3Config:
    config = UR3Config(**json.loads(Path(path).read_text(encoding="utf-8")))
    config.validate()
    return config


class DecisionState:
    """Single-owner decision state; the controller protects it with a lock."""

    def __init__(self, config: UR3Config, now: float):
        self.config = config
        self.last_frame = now
        self.candidate = None
        self.count = 0
        self.direction = 0
        self.reason = "STARTUP"

    def check_fresh(self, now: float) -> None:
        if now - self.last_frame >= self.config.inference_timeout:
            self.direction = 0
            self.reason = "INFERENCE TIMEOUT"
            raise RuntimeError("UR3: 判定更新が途絶えました（INFERENCE TIMEOUT）")

    def update(self, verdict: str, frame_started: float, now: float) -> None:
        self.check_fresh(now)
        if not math.isfinite(frame_started) or not self.last_frame <= frame_started <= now:
            raise ValueError("UR3: フレーム時刻が不正です")
        if now - frame_started >= self.config.inference_timeout:
            self.direction = 0
            self.reason = "INFERENCE TIMEOUT"
            raise RuntimeError("UR3: フレームの判定が古すぎます（INFERENCE TIMEOUT）")
        if verdict not in ("NORMAL", "ANOMALY", "NO OBJECT", "INVALID"):
            raise ValueError(f"UR3: 不明な判定 {verdict!r}")
        self.last_frame = frame_started
        if verdict in ("NO OBJECT", "INVALID"):
            self.candidate, self.count, self.direction = None, 0, 0
            self.reason = verdict
            return
        if self.candidate != verdict:
            self.candidate, self.count, self.direction = verdict, 0, 0
        self.count = min(self.count + 1, self.config.stable_count)
        self.reason = f"CONFIRM {verdict}"
        if self.count == self.config.stable_count:
            self.direction = (
                self.config.normal_direction if verdict == "NORMAL"
                else self.config.anomaly_direction
            )
            self.reason = verdict


def control_script(config: UR3Config) -> str:
    """Target CB3 PolyScope 3.5.1; socket reads accept only two arguments.

    The native read timeout is fixed at two seconds. A sync-based watchdog
    enforces communication_timeout without waiting for that read to return.
    """
    config.validate()
    watchdog_ticks = math.ceil(config.communication_timeout / CB3_CYCLE_SECONDS)
    return f'''def ingot_wrist_control():
  global ingot_alive = True
  global ingot_direction = 0
  global ingot_stop_requested = False
  global ingot_lease_ticks = 0
  thread ingot_watchdog():
    while ingot_alive:
      sync()
      enter_critical
      if ingot_alive:
        ingot_lease_ticks = ingot_lease_ticks + 1
        if ingot_lease_ticks >= {watchdog_ticks}:
          ingot_direction = 0
          ingot_alive = False
        end
      end
      exit_critical
    end
  end
  thread ingot_receive():
    local sequence = 0
    while ingot_alive:
      local packet = socket_read_ascii_float(2, "ingot")
      local valid = False
      if packet[0] == 2:
        if packet[1] == sequence + 1:
          if packet[2] == -1 or packet[2] == 0 or packet[2] == 1 or packet[2] == 9:
            valid = True
          end
        end
      end
      if not valid:
        enter_critical
        ingot_direction = 0
        ingot_alive = False
        exit_critical
      elif packet[2] == 9:
        enter_critical
        ingot_direction = 0
        ingot_alive = False
        exit_critical
      else:
        local accepted = False
        enter_critical
        if ingot_alive:
          sequence = sequence + 1
          ingot_lease_ticks = 0
          ingot_direction = packet[2]
          if packet[2] == 0:
            ingot_stop_requested = True
          end
          accepted = True
        end
        exit_critical
        if accepted:
          if not socket_send_int(sequence, "ingot"):
            enter_critical
            ingot_direction = 0
            ingot_alive = False
            exit_critical
          end
        end
      end
    end
  end
  if socket_open("{config.listen_host}", {config.listen_port}, "ingot"):
    local receiver = run ingot_receive()
    local watchdog = run ingot_watchdog()
    local ready = socket_send_line("READY ingot-v1", "ingot")
    if ready:
      local active = 0
      local running = True
      while running:
        enter_critical
        running = ingot_alive
        local target = ingot_direction
        local must_stop = ingot_stop_requested
        exit_critical
        if active != 0:
          if not running or must_stop or target != active:
            stopj({config.stop_deceleration})
            active = 0
          end
        end
        # Only clear the stop latch while stationary. A zero command must not
        # disappear if another direction arrives during a speedj time slice.
        if active == 0:
          enter_critical
          running = ingot_alive
          target = ingot_direction
          ingot_stop_requested = False
          exit_critical
        end
        if running:
          if target != 0:
            local velocity = target * {config.speed}
            speedj([0, 0, 0, 0, 0, velocity], {config.acceleration}, {config.motion_period})
            active = target
          else:
            sync()
          end
        end
      end
      stopj({config.stop_deceleration})
    else:
      enter_critical
      ingot_direction = 0
      ingot_alive = False
      exit_critical
    end
    kill receiver
    kill watchdog
    if ready:
      socket_send_line("STOPPED", "ingot")
    end
    socket_close("ingot")
  end
end
'''


class RobotChannel:
    """One outstanding command; ACKs are signed, network-order 32-bit sequences."""

    def __init__(self, connection: socket.socket, config: UR3Config):
        self.connection, self.config = connection, config
        self.sequence = 0

    def _read(self, size: int, deadline: float) -> bytes:
        result = bytearray()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("UR3: 応答タイムアウト")
            self.connection.settimeout(remaining)
            chunk = self.connection.recv(size - len(result))
            if not chunk:
                raise ConnectionError("UR3: 接続が切断されました")
            result.extend(chunk)
        return bytes(result)

    def _line(self, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        result = bytearray()
        while len(result) < 128:
            result.extend(self._read(1, deadline))
            if result.endswith(b"\n"):
                return bytes(result)
        raise RuntimeError("UR3: 応答行が長すぎます")

    def ready(self) -> None:
        if self._line(self.config.connect_timeout) != b"READY ingot-v1\n":
            raise RuntimeError("UR3: 準備完了応答が不正です")

    def _send(self, direction: int) -> None:
        self.sequence += 1
        if self.sequence > 2**31 - 1:
            raise RuntimeError("UR3: 通信連番の上限です。再起動してください")
        self.connection.settimeout(self.config.communication_timeout)
        self.connection.sendall(f"({self.sequence},{direction})\n".encode("ascii"))

    def command(self, direction: int) -> None:
        if direction not in (-1, 0, 1):
            raise ValueError("UR3: 回転指令が不正です")
        self._send(direction)
        reply = self._read(4, time.monotonic() + self.config.communication_timeout)
        if struct.unpack("!i", reply)[0] != self.sequence:
            raise RuntimeError("UR3: ACK連番が不正です")

    def finish(self) -> None:
        self._send(9)
        if self._line(self.config.shutdown_timeout) != b"STOPPED\n":
            raise RuntimeError("UR3: 停止完了応答が不正です")

    def close(self) -> None:
        self.connection.close()


class WristController:
    """Own the transport and enforce inference freshness outside the GUI loop."""

    def __init__(self, config: UR3Config, *, dry_run=False, clock=time.monotonic):
        config.validate()
        self.config, self.dry_run, self.clock = config, dry_run, clock
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._wake = threading.Event()
        self._worker = self._drainer = None
        self._primary = self._listener = self._channel = None
        self._failure = None
        self._state = None
        # Preserve a brief stop even if more than three new frames arrive
        # before the communication worker can send the next command.
        self._stop_epoch = 0
        self._acked_stop_epoch = -1
        self.stopped_confirmed = False

    def _log(self, message: str) -> None:
        mode = "DRY-RUN" if self.dry_run else "UR3"
        print(f"[{mode}] {message}", flush=True)

    def _drain_primary(self) -> None:
        # Port 30002 continuously publishes state. Drain it while keeping the
        # upload connection alive; never interpret a TCP send as robot motion.
        try:
            while not self._closing.is_set():
                try:
                    if not self._primary.recv(4096):
                        raise ConnectionError("UR3: 30002接続が切断されました")
                except socket.timeout:
                    continue
        except OSError as error:
            if not self._closing.is_set():
                with self._lock:
                    self._failure = error

    def _open(self) -> None:
        c = self.config
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((c.listen_host, c.listen_port))
        self._listener.listen(1)
        self._primary = socket.create_connection((c.robot_host, c.robot_port), c.connect_timeout)
        self._primary.sendall(control_script(c).encode("ascii"))
        self._primary.settimeout(0.2)
        self._drainer = threading.Thread(target=self._drain_primary, daemon=True)
        self._drainer.start()
        deadline = time.monotonic() + c.connect_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("UR3: Jetsonへの接続待ちがタイムアウトしました")
            self._listener.settimeout(remaining)
            connection, peer = self._listener.accept()
            if peer[0] == c.robot_host:
                break
            connection.close()
        self._channel = RobotChannel(connection, c)
        self._listener.close()
        self._listener = None
        self._channel.ready()

    def __enter__(self):
        try:
            if not self.dry_run:
                self._open()
            with self._lock:
                self._state = DecisionState(self.config, self.clock())
            self._log("DRY-RUN（実機接続なし）" if self.dry_run else "接続・準備完了（回転指令=0）")
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
            self.check()
            return self
        except BaseException:
            self.close()
            raise

    def publish(self, verdict: str, frame_started: float) -> None:
        with self._lock:
            self._check_locked()
            try:
                previous_direction = self._state.direction
                self._state.update(verdict, frame_started, self.clock())
                if previous_direction != 0 and self._state.direction == 0:
                    self._stop_epoch += 1
                    self._wake.set()
            except Exception as error:
                self._state.direction = 0
                self._failure = error
                raise

    def _check_locked(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"UR3制御を終了しました: {self._failure}") from self._failure

    def check(self) -> None:
        with self._lock:
            self._check_locked()

    def status(self) -> str:
        with self._lock:
            s = self._state
            mode = "DRY-RUN" if self.dry_run else "CONNECTED"
            if self._failure:
                mode = "ERROR"
            return (
                f"UR3 {mode} cmd={s.direction:+d} "
                f"count={s.count}/{self.config.stable_count} {s.reason}"
            )

    def _run(self) -> None:
        healthy = True
        previous = None
        try:
            while not self._closing.is_set():
                self._wake.clear()
                started = time.monotonic()
                with self._lock:
                    self._check_locked()
                    self._state.check_fresh(self.clock())
                    direction, reason = self._state.direction, self._state.reason
                    stop_epoch = self._stop_epoch
                    if self._acked_stop_epoch < stop_epoch:
                        if direction != 0:
                            reason = "STOP BEFORE UPDATE"
                        direction = 0
                if self._channel is not None:
                    try:
                        self._channel.command(direction)
                    except Exception:
                        healthy = False  # Outstanding reply is unknown: do not reuse the stream.
                        raise
                if direction == 0:
                    with self._lock:
                        self._acked_stop_epoch = stop_epoch
                if previous != (direction, reason):
                    self._log(f"回転指令={direction:+d} 理由={reason}（実回転は未確認）")
                    previous = direction, reason
                delay = self.config.heartbeat_interval - (time.monotonic() - started)
                self._wake.wait(max(0, delay))
        except Exception as error:
            with self._lock:
                self._failure = error
                self._state.direction = 0
            self._log(f"停止要求: {error}")
        finally:
            if self._channel is not None:
                try:
                    if healthy:
                        self._channel.finish()
                        self.stopped_confirmed = True
                except Exception as error:
                    with self._lock:
                        self._failure = RuntimeError(f"停止未確認: {error}")
                finally:
                    self._channel.close()
            else:
                self.stopped_confirmed = self.dry_run
            with self._lock:
                self._state.direction = 0
            if self.dry_run:
                self._log("回転指令=0 終了（実機接続なし）")
            elif self.stopped_confirmed:
                self._log("停止完了応答を受信しました")
            else:
                self._log("停止未確認: 接続を終了しました。UR3側の通信監視で停止します")

    def close(self) -> None:
        self._closing.set()
        self._wake.set()
        if self._worker is not None:
            limit = 2 * self.config.communication_timeout + self.config.shutdown_timeout + 1
            self._worker.join(limit)
            if self._worker.is_alive():
                if self._channel is not None:
                    self._channel.close()
                with self._lock:
                    self._failure = RuntimeError("UR3: 通信処理を終了できません（停止未確認）")
                self._log(str(self._failure))
        elif self._channel is not None:
            self._channel.close()
            self._log("起動失敗・接続終了（停止未確認、回転指令は送信していません）")
        for connection in (self._listener, self._primary):
            if connection is not None:
                connection.close()
        if self._drainer is not None:
            self._drainer.join(0.5)

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        if exc_type is None:
            self.check()
