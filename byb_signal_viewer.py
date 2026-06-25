"""
AdaptStim — Adaptive Peroneal Nerve Stimulation Prototype

Modes:
    Manual Mode     — supervised device testing and debugging
    Smart Mode      — automated stimulation/rest protocol

Run:
    python byb_signal_viewer.py --port COM4 --esp-port COM5

Web app: http://127.0.0.1:8060
"""

from __future__ import annotations

import argparse
import logging
import math
import socket
import sys
import threading
import time
import webbrowser
from collections import deque
from typing import Optional

import numpy as np
from scipy import signal

try:
    import serial
    from serial import SerialException
except ImportError as exc:
    raise SystemExit("pyserial is required: pip install pyserial") from exc

try:
    import dash
    from dash import Input, Output, State, dcc, html, callback_context
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit("dash and plotly are required: pip install dash plotly") from exc


# ── Hardware ────────────────────────────────────────────────────────────────
DEFAULT_BYB_PORT = "COM4"
DEFAULT_BAUD = 222_222
DEFAULT_ESP32_PORT = "COM5"
DEFAULT_ESP32_BAUD = 230_400
DEFAULT_WEB_PORT = 8060
READ_CHUNK_SIZE = 8192
SERIAL_TIMEOUT_SECONDS = 0.02
MAX_BUFFER_SECONDS = 45.0
MAX_BUFFER_SAMPLES = 350_000
MAX_PLOT_POINTS = 2600

# ── Signal processing ───────────────────────────────────────────────────────
NOMINAL_SAMPLE_RATE_HZ = 10_000.0
EMG_HIGHPASS_HZ = 20.0
EMG_LOWPASS_HZ = 450.0
NOTCH_HZ = 60.0
NOTCH_Q = 30.0
ENVELOPE_LOWPASS_HZ = 8.0

# ── Measurement (contract/relax) ────────────────────────────────────────────
MEAS_READY_SECONDS = 3.0
MEAS_CONTRACT_SECONDS = 5.0
MEAS_RELAX_SECONDS = 5.0
MEAS_DURATION_SECONDS = 30.0
MIN_VALID_WINDOWS = 2
MIN_RMS_SIGNAL = 1.0
MDF_VALID_MIN_HZ = 20.0
MDF_VALID_MAX_HZ = 480.0
CLIP_RATIO_MAX = 0.15        # fraction of samples at rail before flagging clip

# ── Smart Mode timing ───────────────────────────────────────────────────────
GET_READY_SECONDS = 10.0
STIM_BLOCK_SECONDS = 300.0   # 5 min
SMART_MAX_CYCLES = 3
POST_STIM_WAIT_SECONDS = 30.0
DEMO_STIM_BLOCK_SECONDS = 60.0
DEMO_POST_STIM_WAIT_SECONDS = 10.0
BREAK_SHORT = 60.0
BREAK_MED = 120.0
BREAK_LONG = 180.0

# ── Stop thresholds (absolute vs first baseline) ────────────────────────────
STOP_RMS_ABS_HIGH = 6.0
STOP_MDF_ABS_LOW = 0.70
STOP_RMS_ABS_COMBO = 5.0
STOP_MDF_ABS_COMBO = 0.80

# ── Break thresholds (rolling vs previous recovery) ─────────────────────────
BREAK3_RMS_ROLL = 2.0
BREAK3_MDF_DROP = 0.05
BREAK3_MDF_DROP_HIGH = 0.10
BREAK3_RMS_ROLL_HIGH = 3.0
BREAK2_RMS_LOW = 1.2
BREAK2_RMS_HIGH = 1.5
BREAK2_MDF_DROP_LOW = 0.03
BREAK2_MDF_DROP_HIGH = 0.10
BREAK1_RMS_MAX = 1.2
BREAK1_MDF_DROP_MAX = 0.03

# ── ESP32 serial semantics ──────────────────────────────────────────────────
# Match esp32_action.ino: serial "ON" energizes the relay, "OFF" de-energizes it.
_SERIAL_STIM_ON = "ON"
_SERIAL_STIM_OFF = "OFF"


# ────────────────────────────────────────────────────────────────────────────
# Hardware: BYB reader
# ────────────────────────────────────────────────────────────────────────────

class BYBSampleReader:
    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.msb: Optional[int] = None

    def clear(self):
        self.msb = None
        try:
            self.serial_port.reset_input_buffer()
        except SerialException:
            pass

    def read_available_samples(self, max_bytes=READ_CHUNK_SIZE) -> list[int]:
        waiting = self.serial_port.in_waiting
        byte_count = min(waiting, max_bytes) if waiting > 0 else max_bytes
        byte_data = self.serial_port.read(byte_count)
        samples: list[int] = []
        for bv in byte_data:
            if bv & 0x80:
                self.msb = bv & 0x7F
                continue
            if self.msb is None:
                continue
            raw = (self.msb << 7) | (bv & 0x7F)
            self.msb = None
            samples.append(raw - 512)
        return samples


# ────────────────────────────────────────────────────────────────────────────
# Signal conditioning (display only — NOT used for RMS/MDF)
# ────────────────────────────────────────────────────────────────────────────

class EMGConditioner:
    def __init__(self, sample_rate_hz: float = NOMINAL_SAMPLE_RATE_HZ):
        self.sample_rate_hz = sample_rate_hz
        nyq = sample_rate_hz / 2.0
        self._notch_b, self._notch_a = signal.iirnotch(NOTCH_HZ / nyq, NOTCH_Q)
        self._bp_sos = signal.butter(4, [EMG_HIGHPASS_HZ / nyq, EMG_LOWPASS_HZ / nyq],
                                     btype="bandpass", output="sos")
        self._env_sos = signal.butter(2, ENVELOPE_LOWPASS_HZ / nyq,
                                      btype="lowpass", output="sos")
        self.reset()

    def reset(self):
        self._notch_zi = signal.lfilter_zi(self._notch_b, self._notch_a) * 0.0
        self._bp_zi = signal.sosfilt_zi(self._bp_sos) * 0.0
        self._env_zi = signal.sosfilt_zi(self._env_sos) * 0.0

    def process(self, samples: list[int]) -> tuple[list[float], list[float]]:
        if not samples:
            return [], []
        data = np.asarray(samples, dtype=float)
        notched, self._notch_zi = signal.lfilter(self._notch_b, self._notch_a,
                                                  data, zi=self._notch_zi)
        filtered, self._bp_zi = signal.sosfilt(self._bp_sos, notched, zi=self._bp_zi)
        envelope, self._env_zi = signal.sosfilt(self._env_sos, np.abs(filtered),
                                                 zi=self._env_zi)
        return filtered.tolist(), envelope.tolist()


# ────────────────────────────────────────────────────────────────────────────
# ESP32 controller
# ────────────────────────────────────────────────────────────────────────────

class ESP32Controller:
    def __init__(self, port_name: str, baud_rate: int):
        self.port_name = port_name
        self.baud_rate = baud_rate
        self._lock = threading.Lock()
        self._port = None
        self._last_cmd = ""
        self._last_msg = f"Ready ({port_name})"
        self._last_ok: Optional[bool] = None
        self._last_time: Optional[str] = None

    def send(self, command: str) -> dict:
        command = command.strip().upper()
        if command not in {"ON", "OFF"}:
            return self._record(command, False, f"Invalid command: {command}")
        with self._lock:
            try:
                port = self._open_locked()
                port.write(f"{command}\n".encode("ascii"))
                port.flush()
                time.sleep(0.15)
                resp = self._read_locked()
                msg = f"Sent {command} to {self.port_name}"
                if resp:
                    msg = f"{msg}: {resp}"
                return self._record(command, True, msg)
            except Exception as exc:
                self._close_locked()
                return self._record(command, False, f"{self.port_name}: {exc}")

    def stim_on(self) -> dict:
        return self.send(_SERIAL_STIM_ON)

    def stim_off(self) -> dict:
        return self.send(_SERIAL_STIM_OFF)

    def release(self) -> dict:
        with self._lock:
            self._close_locked()
            return self._record("RELEASE", True, f"Released {self.port_name}")

    def _open_locked(self):
        if self._port is not None and self._port.is_open:
            return self._port
        self._port = serial.Serial(self.port_name, self.baud_rate,
                                   timeout=0.5, write_timeout=0.5)
        time.sleep(1.5)
        try:
            self._port.reset_input_buffer()
        except SerialException:
            pass
        return self._port

    def _read_locked(self) -> str:
        if not self._port or not self._port.is_open:
            return ""
        w = getattr(self._port, "in_waiting", 0)
        if not w:
            return ""
        return self._port.read(w).decode("utf-8", errors="replace").strip()

    def _close_locked(self):
        if self._port is None:
            return
        try:
            if self._port.is_open:
                self._port.close()
        finally:
            self._port = None

    def _record(self, cmd, ok, msg) -> dict:
        self._last_cmd = cmd
        self._last_ok = ok
        self._last_msg = msg
        self._last_time = time.strftime("%H:%M:%S")
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "connected": bool(self._port and self._port.is_open),
            "port": self.port_name,
            "last_cmd": self._last_cmd,
            "last_msg": self._last_msg,
            "last_ok": self._last_ok,
            "last_time": self._last_time,
        }


# ────────────────────────────────────────────────────────────────────────────
# Protocol state constants
# ────────────────────────────────────────────────────────────────────────────

class AppState:
    IDLE = "IDLE"
    MANUAL = "MANUAL"
    FULL_GET_READY = "FULL_GET_READY"
    INITIAL_BASELINE = "INITIAL_BASELINE"
    STIMULATING = "STIMULATING"
    POST_STIM_WAIT = "POST_STIM_WAIT"
    RECOVERY_MEASUREMENT = "RECOVERY_MEASUREMENT"
    BREAK = "BREAK"
    COMPLETE = "COMPLETE"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"
    ERROR = "ERROR"

class CueState:
    READY = "READY"
    CONTRACT = "CONTRACT"
    RELAX = "RELAX"
    DONE = "DONE"


def _cue_at(phase_elapsed: float, duration: float) -> tuple[str, float]:
    """Return (cue_state, seconds_remaining_in_window)."""
    if phase_elapsed >= duration:
        return CueState.DONE, 0.0
    if phase_elapsed < MEAS_READY_SECONDS:
        return CueState.READY, MEAS_READY_SECONDS - phase_elapsed
    cycle = (phase_elapsed - MEAS_READY_SECONDS) % (MEAS_CONTRACT_SECONDS + MEAS_RELAX_SECONDS)
    if cycle < MEAS_CONTRACT_SECONDS:
        return CueState.CONTRACT, MEAS_CONTRACT_SECONDS - cycle
    return CueState.RELAX, (MEAS_CONTRACT_SECONDS + MEAS_RELAX_SECONDS) - cycle


# ────────────────────────────────────────────────────────────────────────────
# Measurement session — contract/relax; RMS/MDF computed here ONLY
# ────────────────────────────────────────────────────────────────────────────

class MeasurementSession:
    def __init__(self, label: str, duration: float = MEAS_DURATION_SECONDS):
        self.label = label          # "initial_baseline" | "recovery" | "manual"
        self.duration = duration
        self._abs_start = time.monotonic()
        self._lock = threading.Lock()
        self._cue = CueState.READY
        self._window_buf: list[float] = []
        self._windows: list[dict] = []   # finalized contraction windows
        self._done = False
        self._sample_rate: Optional[float] = None

    # Called from the serial thread
    def push(self, filtered: list[float], sample_rate: float):
        if not filtered:
            return
        elapsed = time.monotonic() - self._abs_start
        new_cue, _ = _cue_at(elapsed, self.duration)
        with self._lock:
            if new_cue == CueState.DONE:
                if self._cue == CueState.CONTRACT and self._window_buf:
                    self._finalize_window(sample_rate)
                self._done = True
                self._cue = CueState.DONE
                return
            if new_cue != self._cue:
                if self._cue == CueState.CONTRACT and self._window_buf:
                    self._finalize_window(sample_rate)
                if new_cue == CueState.CONTRACT:
                    self._window_buf = []
                self._cue = new_cue
            if self._cue == CueState.CONTRACT:
                self._window_buf.extend(filtered)
                self._sample_rate = sample_rate

    def _finalize_window(self, sr: float):
        data = np.asarray(self._window_buf, dtype=float)
        data = data - float(np.mean(data))
        rms = float(np.sqrt(np.mean(data ** 2)))
        mdf = _compute_mdf(data, sr)
        clipped = float(np.mean(np.abs(data) >= 510))
        self._windows.append({"rms": rms, "mdf": mdf, "clip_ratio": clipped, "n": len(data)})
        self._window_buf = []

    def is_complete(self) -> bool:
        return time.monotonic() - self._abs_start >= self.duration

    def elapsed(self) -> float:
        return min(time.monotonic() - self._abs_start, self.duration)

    def cue_snapshot(self) -> dict:
        e = self.elapsed()
        cue, remaining = _cue_at(e, self.duration)
        return {"cue": cue, "remaining": remaining, "elapsed": e, "duration": self.duration}

    def result(self) -> dict:
        with self._lock:
            windows = list(self._windows)
        valid = [w for w in windows
                 if w["rms"] > MIN_RMS_SIGNAL
                 and w["mdf"] is not None
                 and MDF_VALID_MIN_HZ <= w["mdf"] <= MDF_VALID_MAX_HZ
                 and w["clip_ratio"] < CLIP_RATIO_MAX]
        rms = float(np.median([w["rms"] for w in valid])) if valid else None
        mdf = float(np.median([w["mdf"] for w in valid])) if valid else None
        quality_ok, quality_reason = _check_quality(valid, rms, mdf)
        return {
            "type": self.label,
            "timestamp": time.strftime("%H:%M:%S"),
            "rms": rms,
            "mdf": mdf,
            "n_windows": len(valid),
            "quality_ok": quality_ok,
            "quality_reason": quality_reason,
        }


def _compute_mdf(data: np.ndarray, sr: float) -> Optional[float]:
    if sr is None or sr <= 0 or len(data) < 64:
        return None
    nperseg = min(len(data), max(64, int(sr)))
    freqs, pwr = signal.welch(data, fs=sr, nperseg=nperseg)
    band = (freqs >= EMG_HIGHPASS_HZ) & (freqs <= EMG_LOWPASS_HZ)
    bf, bp = freqs[band], pwr[band]
    total = float(np.sum(bp))
    if bf.size == 0 or total <= 0:
        return None
    idx = int(np.searchsorted(np.cumsum(bp), total * 0.5))
    idx = min(idx, bf.size - 1)
    v = float(bf[idx])
    return v if math.isfinite(v) else None


def _check_quality(valid_windows: list, rms: Optional[float],
                   mdf: Optional[float]) -> tuple[bool, str]:
    if len(valid_windows) < MIN_VALID_WINDOWS:
        return False, f"Only {len(valid_windows)} valid contraction window(s); need {MIN_VALID_WINDOWS}."
    if rms is None or rms <= MIN_RMS_SIGNAL:
        return False, "RMS near zero — check electrode contact."
    if mdf is None:
        return False, "MDF could not be computed — signal too short or flat."
    return True, "OK"


def _check_stop(rms_abs: float, mdf_abs: float) -> tuple[bool, str]:
    if rms_abs >= STOP_RMS_ABS_HIGH:
        return True, f"RMS_abs {rms_abs:.2f} ≥ {STOP_RMS_ABS_HIGH} (high activation)"
    if mdf_abs <= STOP_MDF_ABS_LOW:
        return True, f"MDF_abs {mdf_abs:.2f} ≤ {STOP_MDF_ABS_LOW} (significant fatigue)"
    if rms_abs >= STOP_RMS_ABS_COMBO and mdf_abs <= STOP_MDF_ABS_COMBO:
        return True, f"RMS_abs {rms_abs:.2f} + MDF_abs {mdf_abs:.2f} combined stop"
    return False, ""


def _assign_break(rms_roll: float, mdf_roll: float) -> float:
    mdf_drop = 1.0 - mdf_roll
    if (rms_roll >= BREAK3_RMS_ROLL and mdf_drop >= BREAK3_MDF_DROP) \
            or mdf_drop >= BREAK3_MDF_DROP_HIGH \
            or rms_roll >= BREAK3_RMS_ROLL_HIGH:
        return BREAK_LONG
    if (BREAK2_RMS_LOW <= rms_roll <= BREAK2_RMS_HIGH) \
            or (BREAK2_MDF_DROP_LOW <= mdf_drop < BREAK2_MDF_DROP_HIGH):
        return BREAK_MED
    if rms_roll < BREAK1_RMS_MAX and mdf_drop < BREAK1_MDF_DROP_MAX:
        return BREAK_SHORT
    return BREAK_MED   # conservative fallback


# ────────────────────────────────────────────────────────────────────────────
# AdaptStim Backend — serial loop + state machine
# ────────────────────────────────────────────────────────────────────────────

class AdaptStimBackend:
    def __init__(self, byb_port: str, byb_baud: int, esp32: ESP32Controller):
        self._byb_port = byb_port
        self._byb_baud = byb_baud
        self.esp32 = esp32

        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._stop_event = threading.Event()

        # Display buffer (for live trace only — no RMS/MDF computed here)
        self._disp_times: deque = deque(maxlen=MAX_BUFFER_SAMPLES)
        self._disp_filtered: deque = deque(maxlen=MAX_BUFFER_SAMPLES)
        self._disp_raw: deque = deque(maxlen=MAX_BUFFER_SAMPLES)
        self._sample_rate = 0.0
        self._total_samples = 0
        self._sample_age: Optional[float] = None
        self._connected = False
        self._byb_msg = "Starting…"
        self._byb_error = ""

        # Protocol state
        self._state = AppState.IDLE
        self._phase_start: float = time.monotonic()
        self._cycle_num = 0
        self._break_seconds = BREAK_MED
        self._stop_reason: Optional[str] = None
        self._error_msg: Optional[str] = None
        self._status_msg: str = "Select a mode to begin."
        self._protocol_mode = "smart"

        # Measurement data
        self._measurement: Optional[MeasurementSession] = None
        self._first_baseline_rms: Optional[float] = None
        self._first_baseline_mdf: Optional[float] = None
        self._prev_recovery_rms: Optional[float] = None
        self._prev_recovery_mdf: Optional[float] = None
        self._last_result: Optional[dict] = None
        self._measurements_log: list[dict] = []

        # Manual measurement support
        self._manual_measurement: Optional[MeasurementSession] = None
        self._manual_result: Optional[dict] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        threading.Thread(target=self._serial_loop, daemon=True).start()
        threading.Thread(target=self._state_machine_loop, daemon=True).start()

    def stop(self):
        self._stop_event.set()
        self.esp32.stim_off()

    # ── Public controls ──────────────────────────────────────────────────────

    def emergency_stop(self):
        threading.Thread(target=self.esp32.stim_off, daemon=True).start()
        with self._lock:
            self._measurement = None
            self._manual_measurement = None
            self._state = AppState.EMERGENCY_STOPPED
            self._status_msg = "EMERGENCY STOP — stimulation halted. Reset to continue."

    def start_full_device(self):
        self._start_protocol("smart")

    def start_demo(self):
        self._start_protocol("demo")

    def _start_protocol(self, mode: str):
        with self._lock:
            if self._state in (AppState.EMERGENCY_STOPPED,):
                return
            self._cycle_num = 0
            self._protocol_mode = mode
            self._first_baseline_rms = None
            self._first_baseline_mdf = None
            self._prev_recovery_rms = None
            self._prev_recovery_mdf = None
            self._last_result = None
            self._measurements_log = []
            self._stop_reason = None
            self._error_msg = None
            self._measurement = None
            self._state = AppState.FULL_GET_READY
            self._phase_start = time.monotonic()
            label = "Demo" if mode == "demo" else "Smart Mode"
            self._status_msg = f"{label} starting in 10 s..."

    def stop_full_device(self):
        threading.Thread(target=self.esp32.stim_off, daemon=True).start()
        with self._lock:
            self._measurement = None
            self._state = AppState.IDLE
            label = "Demo" if self._protocol_mode == "demo" else "Smart Mode"
            self._status_msg = f"{label} stopped by user."

    def enter_manual(self):
        with self._lock:
            if self._state not in (AppState.IDLE, AppState.MANUAL):
                return
            self._state = AppState.MANUAL
            self._status_msg = "Manual Mode active."

    def exit_manual(self):
        with self._lock:
            if self._state == AppState.MANUAL:
                self._state = AppState.IDLE
                self._manual_measurement = None
                self._status_msg = "Returned to Home."

    def reset_from_error(self):
        threading.Thread(target=self.esp32.stim_off, daemon=True).start()
        with self._lock:
            self._measurement = None
            self._state = AppState.IDLE
            self._error_msg = None
            self._status_msg = "Reset. Select a mode to begin."

    def start_manual_measurement(self):
        with self._lock:
            if self._state != AppState.MANUAL:
                return
            self._manual_measurement = MeasurementSession("manual", MEAS_DURATION_SECONDS)
            self._manual_result = None
            self._status_msg = "Manual measurement started — follow the cue."

    # ── Serial loop ──────────────────────────────────────────────────────────

    def _serial_loop(self):
        port = None
        reader = None
        conditioner = EMGConditioner()
        last_batch_t = time.monotonic()
        last_rate_t = time.monotonic()
        last_rate_count = 0

        while not self._stop_event.is_set():
            if port is None:
                try:
                    self._set_byb_state(False, f"Opening {self._byb_port}…")
                    port = serial.Serial(self._byb_port, self._byb_baud,
                                        timeout=SERIAL_TIMEOUT_SECONDS)
                    time.sleep(0.1)
                    reader = BYBSampleReader(port)
                    reader.clear()
                    conditioner.reset()
                    last_batch_t = time.monotonic()
                    self._set_byb_state(True, f"Streaming from {self._byb_port}")
                except SerialException as e:
                    self._set_byb_state(False, f"Waiting for {self._byb_port}", str(e))
                    time.sleep(1.0)
                    continue

            try:
                samples = reader.read_available_samples()
            except (OSError, SerialException) as e:
                try:
                    port.close()
                except Exception:
                    pass
                port = None
                reader = None
                self._set_byb_state(False, f"Lost {self._byb_port}", str(e))
                time.sleep(0.5)
                continue

            if samples:
                filtered, _ = conditioner.process(samples)
                now = time.monotonic()
                span = max(now - last_batch_t, 1e-9)
                interval = span / len(samples)
                elapsed_now = now - self._start_time
                last_batch_t = now

                with self._lock:
                    for i, s in enumerate(samples):
                        t = elapsed_now - (len(samples) - i - 1) * interval
                        self._disp_times.append(t)
                        self._disp_raw.append(s)
                        self._disp_filtered.append(filtered[i])
                    self._total_samples += len(samples)
                    cutoff = elapsed_now - MAX_BUFFER_SECONDS
                    while self._disp_times and self._disp_times[0] < cutoff:
                        self._disp_times.popleft()
                        self._disp_raw.popleft()
                        self._disp_filtered.popleft()
                    self._sample_age = 0.0
                    meas = self._measurement
                    man_meas = self._manual_measurement

                sr = self._sample_rate if self._sample_rate > 0 else NOMINAL_SAMPLE_RATE_HZ
                if meas is not None:
                    meas.push(filtered, sr)
                if man_meas is not None:
                    man_meas.push(filtered, sr)

            now = time.monotonic()
            if now - last_rate_t >= 1.0:
                with self._lock:
                    total = self._total_samples
                    age = self._sample_age
                    if age is not None:
                        self._sample_age = age + (now - last_rate_t)
                self._sample_rate = (total - last_rate_count) / (now - last_rate_t)
                last_rate_count = total
                last_rate_t = now

    def _set_byb_state(self, connected: bool, msg: str, error: str = ""):
        with self._lock:
            self._connected = connected
            self._byb_msg = msg
            self._byb_error = error

    # ── State machine loop ───────────────────────────────────────────────────

    def _state_machine_loop(self):
        while not self._stop_event.is_set():
            time.sleep(0.5)
            try:
                self._tick()
            except Exception as e:
                with self._lock:
                    self._state = AppState.ERROR
                    self._error_msg = f"Internal error: {e}"

    def _stim_block_seconds_locked(self) -> float:
        return DEMO_STIM_BLOCK_SECONDS if self._protocol_mode == "demo" else STIM_BLOCK_SECONDS

    def _post_stim_wait_seconds_locked(self) -> float:
        return DEMO_POST_STIM_WAIT_SECONDS if self._protocol_mode == "demo" else POST_STIM_WAIT_SECONDS

    def _tick(self):
        with self._lock:
            state = self._state
            elapsed = time.monotonic() - self._phase_start
            meas = self._measurement
            man_meas = self._manual_measurement
            stim_seconds = self._stim_block_seconds_locked()
            post_wait_seconds = self._post_stim_wait_seconds_locked()

        if state == AppState.FULL_GET_READY:
            if elapsed >= GET_READY_SECONDS:
                self._begin_measurement("initial_baseline")

        elif state == AppState.INITIAL_BASELINE:
            if meas and meas.is_complete():
                self._finalize_initial_baseline()

        elif state == AppState.STIMULATING:
            if elapsed >= stim_seconds:
                threading.Thread(target=self.esp32.stim_off, daemon=True).start()
                with self._lock:
                    self._state = AppState.POST_STIM_WAIT
                    self._phase_start = time.monotonic()
                    self._status_msg = f"Stimulation complete. Waiting {int(post_wait_seconds)} s before measurement..."

        elif state == AppState.POST_STIM_WAIT:
            if elapsed >= post_wait_seconds:
                self._begin_measurement("recovery")

        elif state == AppState.RECOVERY_MEASUREMENT:
            if meas and meas.is_complete():
                self._finalize_recovery()

        elif state == AppState.BREAK:
            with self._lock:
                bk = self._break_seconds
            if elapsed >= bk:
                self._begin_stim_block()

        elif state == AppState.MANUAL:
            if man_meas and man_meas.is_complete():
                result = man_meas.result()
                with self._lock:
                    self._manual_result = result
                    self._manual_measurement = None
                    self._status_msg = "Manual measurement complete."

    def _begin_measurement(self, label: str):
        meas = MeasurementSession(label, MEAS_DURATION_SECONDS)
        state = AppState.INITIAL_BASELINE if label == "initial_baseline" else AppState.RECOVERY_MEASUREMENT
        with self._lock:
            self._measurement = meas
            self._state = state
            self._phase_start = time.monotonic()
            self._status_msg = f"{'Baseline' if label == 'initial_baseline' else 'Recovery'} measurement — follow the cue."

    def _finalize_initial_baseline(self):
        with self._lock:
            meas = self._measurement
        result = meas.result()
        result["cycle"] = 0
        result["phase"] = "Baseline"
        result["break_assigned"] = None
        result["rms_abs"] = 1.0 if result.get("rms") else None
        result["mdf_abs"] = 1.0 if result.get("mdf") else None
        with self._lock:
            self._last_result = result
            self._measurements_log.append(result)
            self._measurement = None
        if not result["quality_ok"]:
            with self._lock:
                self._state = AppState.ERROR
                self._error_msg = f"Initial baseline quality failed: {result['quality_reason']}"
                self._status_msg = "Signal quality check failed. Check electrodes and retry."
            return
        with self._lock:
            self._first_baseline_rms = result["rms"]
            self._first_baseline_mdf = result["mdf"]
        self._begin_stim_block()

    def _begin_stim_block(self):
        threading.Thread(target=self.esp32.stim_on, daemon=True).start()
        with self._lock:
            self._cycle_num += 1
            mode = self._protocol_mode
            stim_seconds = self._stim_block_seconds_locked()
            self._state = AppState.STIMULATING
            self._phase_start = time.monotonic()
            duration_label = f"{int(stim_seconds // 60)}-minute" if stim_seconds >= 60 else f"{int(stim_seconds)}-second"
            label = "Demo" if mode == "demo" else "Smart Mode"
            self._status_msg = f"{label} stimulation — cycle {self._cycle_num}, {duration_label} block."

    def _finalize_recovery(self):
        with self._lock:
            meas = self._measurement
            first_rms = self._first_baseline_rms
            first_mdf = self._first_baseline_mdf
            prev_rms = self._prev_recovery_rms or first_rms
            prev_mdf = self._prev_recovery_mdf or first_mdf
            cycle = self._cycle_num
            mode = self._protocol_mode
        result = meas.result()
        result["cycle"] = cycle
        result["phase"] = "Recovery"

        if not result["quality_ok"]:
            with self._lock:
                self._last_result = result
                self._measurements_log.append(result)
                self._measurement = None
                self._state = AppState.ERROR
                self._error_msg = f"Recovery measurement quality failed: {result['quality_reason']}"
                self._status_msg = "Signal quality check failed. Check electrodes and retry."
            return

        rms = result["rms"]
        mdf = result["mdf"]
        rms_abs = rms / first_rms if first_rms else 1.0
        mdf_abs = mdf / first_mdf if (first_mdf and mdf) else 1.0
        result["rms_abs"] = round(rms_abs, 3)
        result["mdf_abs"] = round(mdf_abs, 3)

        should_stop, stop_reason = _check_stop(rms_abs, mdf_abs)
        if should_stop:
            result["break_assigned"] = None
            threading.Thread(target=self.esp32.stim_off, daemon=True).start()
            with self._lock:
                self._last_result = result
                self._measurements_log.append(result)
                self._measurement = None
                self._state = AppState.COMPLETE
                self._stop_reason = stop_reason
                self._status_msg = f"Session complete — {stop_reason}"
            return

        rms_roll = rms / prev_rms if prev_rms else 1.0
        mdf_roll = mdf / prev_mdf if (prev_mdf and mdf) else 1.0
        break_s = _assign_break(rms_roll, mdf_roll)
        result["break_assigned"] = int(break_s)

        with self._lock:
            self._last_result = result
            self._measurements_log.append(result)
            self._measurement = None
            self._prev_recovery_rms = rms
            self._prev_recovery_mdf = mdf
            self._break_seconds = break_s
            if mode == "demo":
                self._state = AppState.COMPLETE
                self._stop_reason = f"Demo complete. Assigned break: {int(break_s // 60)} min."
                self._status_msg = self._stop_reason
            elif cycle >= SMART_MAX_CYCLES:
                self._state = AppState.COMPLETE
                self._stop_reason = f"Smart Mode complete after {SMART_MAX_CYCLES} stimulation blocks."
                self._status_msg = self._stop_reason
            else:
                self._state = AppState.BREAK
                self._phase_start = time.monotonic()
                self._status_msg = f"Recovery OK. {int(break_s // 60)}-minute break before next block."

    # ── Snapshot API (called from Dash callbacks) ────────────────────────────

    def snapshot(self, window_seconds: float = 10.0) -> dict:
        now = time.monotonic()
        with self._lock:
            times = list(self._disp_times)
            filtered = list(self._disp_filtered)
            raw = list(self._disp_raw)
            total = self._total_samples
            sr = self._sample_rate
            age = self._sample_age
            connected = self._connected
            msg = self._byb_msg
            error = self._byb_error
            state = self._state
            phase_start = self._phase_start
            cycle = self._cycle_num
            break_s = self._break_seconds
            stop_reason = self._stop_reason
            error_msg = self._error_msg
            status_msg = self._status_msg
            first_rms = self._first_baseline_rms
            first_mdf = self._first_baseline_mdf
            last_result = self._last_result
            meas = self._measurement
            man_meas = self._manual_measurement
            man_result = self._manual_result
            logs = list(self._measurements_log)
            n_logs = len(logs)
            protocol_mode = self._protocol_mode
            stim_seconds = self._stim_block_seconds_locked()
            post_wait_seconds = self._post_stim_wait_seconds_locked()

        phase_elapsed = now - phase_start
        phase_remaining = None
        durations = {
            AppState.FULL_GET_READY: GET_READY_SECONDS,
            AppState.STIMULATING: stim_seconds,
            AppState.POST_STIM_WAIT: post_wait_seconds,
            AppState.BREAK: break_s,
            AppState.INITIAL_BASELINE: MEAS_DURATION_SECONDS,
            AppState.RECOVERY_MEASUREMENT: MEAS_DURATION_SECONDS,
        }
        if state in durations:
            phase_remaining = max(0.0, durations[state] - phase_elapsed)

        meas_cue = meas.cue_snapshot() if meas else None
        man_cue = man_meas.cue_snapshot() if man_meas else None

        # Build plot data
        if times:
            latest_t = times[-1]
            cutoff = latest_t - window_seconds
            start_i = next((i for i, t in enumerate(times) if t >= cutoff), 0)
            wt = times[start_i:]
            wf = filtered[start_i:]
        else:
            latest_t = phase_elapsed
            wt, wf = [], []

        pt, pf = _downsample(wt, wf, MAX_PLOT_POINTS)
        xs = [t - latest_t for t in pt]

        return {
            "x": xs, "y": pf,
            "connected": connected, "byb_msg": msg, "byb_error": error,
            "sample_rate": sr, "total_samples": total, "sample_age": age,
            "state": state, "phase_elapsed": phase_elapsed,
            "phase_remaining": phase_remaining, "cycle": cycle,
            "protocol_mode": protocol_mode, "stim_seconds": stim_seconds,
            "post_wait_seconds": post_wait_seconds,
            "break_seconds": break_s, "stop_reason": stop_reason,
            "error_msg": error_msg, "status_msg": status_msg,
            "first_baseline_rms": first_rms, "first_baseline_mdf": first_mdf,
            "last_result": last_result, "meas_cue": meas_cue,
            "man_cue": man_cue, "manual_result": man_result,
            "n_measurements": n_logs, "measurements_log": logs,
        }


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _downsample(times, samples, max_pts):
    if len(samples) <= max_pts:
        return times, samples
    buckets = max(1, max_pts // 2)
    bsz = max(1, math.ceil(len(samples) / buckets))
    pt, ps = [], []
    for start in range(0, len(samples), bsz):
        end = min(start + bsz, len(samples))
        bucket = samples[start:end]
        btimes = times[start:end]
        if not bucket:
            continue
        lo = min(range(len(bucket)), key=bucket.__getitem__)
        hi = max(range(len(bucket)), key=bucket.__getitem__)
        for idx in sorted({lo, hi}):
            pt.append(btimes[idx])
            ps.append(bucket[idx])
    return pt, ps


def _fmt(v: Optional[float], decimals=1, suffix="") -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{suffix}"


def _fmt_countdown(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    s = int(max(0, seconds))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def _phase_label(state: str) -> str:
    return {
        AppState.IDLE: "Idle",
        AppState.MANUAL: "Manual Mode",
        AppState.FULL_GET_READY: "Get Ready",
        AppState.INITIAL_BASELINE: "Initial Baseline",
        AppState.STIMULATING: "Stimulating",
        AppState.POST_STIM_WAIT: "Rest Wait",
        AppState.RECOVERY_MEASUREMENT: "Recovery Measurement",
        AppState.BREAK: "Break",
        AppState.COMPLETE: "Session Complete",
        AppState.EMERGENCY_STOPPED: "Emergency Stopped",
        AppState.ERROR: "Error",
    }.get(state, state)


def _phase_color(state: str) -> str:
    return {
        AppState.IDLE: "#94a3b8",
        AppState.MANUAL: "#475569",
        AppState.FULL_GET_READY: "#0891b2",
        AppState.INITIAL_BASELINE: "#7c3aed",
        AppState.STIMULATING: "#059669",
        AppState.POST_STIM_WAIT: "#d97706",
        AppState.RECOVERY_MEASUREMENT: "#7c3aed",
        AppState.BREAK: "#0891b2",
        AppState.COMPLETE: "#059669",
        AppState.EMERGENCY_STOPPED: "#dc2626",
        AppState.ERROR: "#dc2626",
    }.get(state, "#94a3b8")


def _cue_color(cue: Optional[str]) -> str:
    return {
        CueState.READY: "#d97706",
        CueState.CONTRACT: "#059669",
        CueState.RELAX: "#0891b2",
        CueState.DONE: "#94a3b8",
    }.get(cue or "", "#94a3b8")


def build_figure(xs, ys, window_seconds: float) -> go.Figure:
    color = "#0891b2"
    trace = go.Scattergl(x=xs, y=ys, mode="lines",
                         line={"color": color, "width": 1.25},
                         hovertemplate="t=%{x:.3f}s<extra></extra>")
    fig = go.Figure(data=[trace])
    fig.update_layout(
        template="plotly_white",
        margin={"l": 52, "r": 16, "t": 12, "b": 44},
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        xaxis={"title": "Seconds from live edge", "range": [-window_seconds, 0],
               "showgrid": True, "gridcolor": "#e2e8f0"},
        yaxis={"title": "Filtered EMG (a.u.)", "showgrid": True, "gridcolor": "#e2e8f0"},
        uirevision="adaptstim-live",
    )
    if not ys:
        fig.add_annotation(text="Waiting for EMG signal…", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font={"size": 16, "color": "#64748b"})
    elif ys:
        y_min, y_max = min(ys), max(ys)
        pad = max(20, (y_max - y_min) * 0.18) if y_max != y_min else 20
        fig.update_yaxes(range=[y_min - pad, y_max + pad])
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Dash application
# ────────────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    background: #f8fafc;
    color: #0f172a;
    min-height: 100vh;
}
.topbar {
    display: flex; align-items: center; gap: 1rem;
    padding: 0 1.25rem; height: 56px;
    background: #ffffff; border-bottom: 1px solid #e2e8f0;
    position: sticky; top: 0; z-index: 100;
}
.logo-text { font-size: 1.3rem; font-weight: 800; color: #0f172a; }
.logo-text span { color: #0891b2; }
.tagline { font-size: 0.75rem; color: #0891b2; font-weight: 600; letter-spacing: .04em; margin-top: 1px; }
.topbar-status { font-size: 0.8rem; color: #64748b; margin-left: auto; display: flex; align-items: center; gap: .5rem; }
.dot { width: .6rem; height: .6rem; border-radius: 50%; display: inline-block; }
.dot-ok { background: #059669; box-shadow: 0 0 0 3px #d1fae5; }
.dot-warn { background: #d97706; box-shadow: 0 0 0 3px #fef3c7; }
.dot-err { background: #dc2626; box-shadow: 0 0 0 3px #fee2e2; }
.estop-btn {
    background: #dc2626; color: #fff; border: none; border-radius: 8px;
    padding: .45rem 1rem; font: inherit; font-weight: 700; font-size: .85rem;
    cursor: pointer; margin-left: .5rem;
}
.estop-btn:hover { background: #b91c1c; }
.shell { display: flex; flex-direction: column; min-height: 100vh; }
.content {
    padding: 1rem clamp(1rem, 2vw, 2rem);
    max-width: 1500px;
    margin: 0 auto;
    width: 100%;
}
.work-grid {
    display: grid;
    grid-template-columns: minmax(360px, .85fr) minmax(520px, 1.25fr);
    gap: 1.1rem;
    align-items: start;
}
.control-stack { min-width: 0; }
.card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 1rem; margin-bottom: .85rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.signal-toolbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: .75rem; margin-bottom: .65rem; flex-wrap: wrap;
}
.signal-window-control {
    display: flex; align-items: center; gap: .5rem;
    font-size: .78rem; font-weight: 600; color: #64748b;
}
.card-title { font-size: .7rem; font-weight: 700; color: #64748b;
              text-transform: uppercase; letter-spacing: .06em; margin-bottom: .65rem; }
.phase-badge {
    display: inline-flex; align-items: center; gap: .5rem;
    padding: .35rem .85rem; border-radius: 999px;
    font-size: .82rem; font-weight: 700; color: #fff;
}
.big-phase {
    font-size: clamp(1.4rem, 3vw, 2.2rem); font-weight: 800;
    line-height: 1.1; margin-bottom: .25rem;
}
.countdown {
    font-size: clamp(1.9rem, 4vw, 3rem); font-weight: 800;
    font-variant-numeric: tabular-nums; color: #0891b2; line-height: 1.05;
}
.countdown:empty { display: none; }
.cue-box {
    border-radius: 10px; padding: 1rem 1.5rem;
    text-align: center; font-size: clamp(1.4rem, 4vw, 2.4rem);
    font-weight: 800; line-height: 1; color: #fff;
    min-height: 4.5rem; display: flex; align-items: center; justify-content: center;
}
.progress-track {
    width: 100%; height: .55rem; background: #e2e8f0;
    border-radius: 999px; overflow: hidden; margin-top: .5rem;
}
.progress-fill { height: 100%; border-radius: 999px; transition: width .4s linear; }
.metric-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: .65rem;
}
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: .6rem .8rem;
}
.metric-label { font-size: .68rem; font-weight: 700; color: #64748b;
                text-transform: uppercase; letter-spacing: .04em; margin-bottom: .2rem; }
.metric-value { font-size: 1.05rem; font-weight: 700; color: #0f172a; }
.readings-scroll { overflow-x: auto; }
.readings-table {
    width: 100%; border-collapse: collapse; font-size: .82rem;
}
.readings-table th {
    text-align: left; font-size: .68rem; text-transform: uppercase;
    letter-spacing: .04em; color: #64748b; background: #f8fafc;
    border-bottom: 1px solid #e2e8f0; padding: .55rem .6rem;
}
.readings-table td {
    border-bottom: 1px solid #eef2f7; padding: .55rem .6rem;
    color: #334155; white-space: nowrap;
}
.readings-table tr:last-child td { border-bottom: none; }
.quality-pill {
    display: inline-flex; align-items: center; border-radius: 999px;
    padding: .15rem .45rem; font-size: .68rem; font-weight: 700;
}
.quality-ok { background: #d1fae5; color: #065f46; }
.quality-warn { background: #fee2e2; color: #991b1b; }
.empty-note { font-size: .82rem; color: #64748b; margin: 0; }
.btn {
    border: 1px solid #cbd5e1; background: #fff; color: #0f172a;
    border-radius: 8px; padding: .45rem .85rem; font: inherit;
    font-weight: 600; font-size: .85rem; cursor: pointer; white-space: nowrap;
}
.btn:hover { background: #f1f5f9; }
.btn:disabled { color: #94a3b8; background: #f8fafc; border-color: #e2e8f0; cursor: default; }
.btn-primary { background: #0891b2; color: #fff; border-color: #0891b2; }
.btn-primary:hover { background: #0e7490; }
.btn-danger { background: #dc2626; color: #fff; border-color: #dc2626; }
.btn-danger:hover { background: #b91c1c; }
.btn-warning { background: #d97706; color: #fff; border-color: #d97706; }
.btn-row { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; }
.section-title { font-size: 1.05rem; font-weight: 700; margin-bottom: .5rem; color: #0f172a; }
.body-text { font-size: .88rem; color: #334155; line-height: 1.65; }
.body-text p { margin-bottom: .75rem; }
.body-text ul { padding-left: 1.25rem; margin-bottom: .75rem; }
.body-text li { margin-bottom: .3rem; }
.warn-box {
    background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px;
    padding: .75rem 1rem; font-size: .85rem; color: #78350f; margin-bottom: .65rem;
}
.danger-box {
    background: #fee2e2; border: 1px solid #fca5a5; border-radius: 8px;
    padding: .75rem 1rem; font-size: .85rem; color: #7f1d1d; margin-bottom: .65rem;
}
.ref-tag { font-size: .75rem; color: #64748b; font-style: italic; }
.view-content {
    flex: 1;
    min-height: calc(100vh - 116px);
}
.plot-frame {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
    overflow: hidden; height: clamp(360px, 54vh, 560px);
}
.dash-graph { height: 100%; }
@media (max-width: 900px) {
    .work-grid { grid-template-columns: 1fr; }
    .plot-frame { height: 300px; }
}
.manual-label {
    display: inline-flex; align-items: center; gap: .4rem;
    background: #fef3c7; border: 1px solid #fcd34d; border-radius: 6px;
    padding: .25rem .65rem; font-size: .75rem; font-weight: 700; color: #92400e;
    margin-bottom: .75rem;
}
.result-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-top: .5rem; }
.result-item { font-size: .9rem; color: #334155; }
.result-item strong { color: #0f172a; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0; }
"""


def _metric(label: str, value_id: str) -> html.Div:
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div("—", id=value_id, className="metric-value"),
    ], className="metric-card")


def _static_metric(label: str, value: str) -> html.Div:
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
    ], className="metric-card")


def _readings_table(logs: list[dict]) -> html.Div:
    if not logs:
        return html.Div([
            html.Div("Cycle Readings", className="card-title"),
            html.P("No baseline or recovery readings recorded yet.", className="empty-note"),
        ], className="card")

    rows = []
    for item in logs:
        phase = item.get("phase") or str(item.get("type", "")).replace("_", " ").title()
        cycle = item.get("cycle", "—")
        break_assigned = item.get("break_assigned")
        break_txt = f"{int(break_assigned // 60)} min" if break_assigned else "—"
        ok = bool(item.get("quality_ok"))
        rows.append(html.Tr([
            html.Td(str(cycle)),
            html.Td(phase),
            html.Td(item.get("timestamp", "—")),
            html.Td(_fmt(item.get("rms"))),
            html.Td(_fmt(item.get("mdf"), suffix=" Hz")),
            html.Td(_fmt(item.get("rms_abs"), decimals=2) if item.get("rms_abs") is not None else "—"),
            html.Td(_fmt(item.get("mdf_abs"), decimals=2) if item.get("mdf_abs") is not None else "—"),
            html.Td(break_txt),
            html.Td(html.Span("OK" if ok else "Check",
                              className=f"quality-pill {'quality-ok' if ok else 'quality-warn'}")),
        ]))

    return html.Div([
        html.Div("Cycle Readings", className="card-title"),
        html.Div(html.Table([
            html.Thead(html.Tr([
                html.Th("Cycle"),
                html.Th("Phase"),
                html.Th("Time"),
                html.Th("RMS"),
                html.Th("MDF"),
                html.Th("RMS x"),
                html.Th("MDF x"),
                html.Th("Break"),
                html.Th("Quality"),
            ])),
            html.Tbody(rows),
        ], className="readings-table"), className="readings-scroll"),
    ], className="card")


def _live_signal_panel(suffix: str) -> html.Div:
    return html.Div([
        html.Div([
            html.Div("Live EMG Signal", className="card-title", style={"marginBottom": 0}),
            html.Div([
                html.Label("Window"),
                dcc.Dropdown(id=f"window-seconds-{suffix}", value=10, clearable=False,
                             options=[{"label": f"{s} s", "value": s} for s in [5, 10, 20, 30]],
                             style={"width": "110px"}),
            ], className="signal-window-control"),
        ], className="signal-toolbar"),
        html.Div(dcc.Graph(id=f"signal-graph-{suffix}",
                           config={"displaylogo": False, "scrollZoom": True, "responsive": True},
                           className="dash-graph"),
                 className="plot-frame"),
    ], className="card")


def _tab_home() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("About AdaptStim", className="section-title"),
            html.Div([
                html.P("AdaptStim is a research-grade adaptive peroneal nerve stimulation prototype. "
                       "It uses surface EMG recordings from a Backyard Brains device to guide a "
                       "rhythmic stimulation and rest cycle delivered by a TENS 7000 unit via an "
                       "ESP32-S3 relay module."),
                html.P("The system monitors muscle response using RMS amplitude and Median Frequency "
                       "(MDF) during defined measurement windows — not continuously during stimulation. "
                       "Based on recovery measurements, it adaptively adjusts rest duration between "
                       "stimulation blocks."),
                html.P("AdaptStim is a prototype for supervised research use only. It is not an "
                       "FDA-cleared medical device and makes no therapeutic claims."),
            ], className="body-text"),
        ], className="card"),
        html.Div([
            html.Div("Modes", className="section-title"),
            html.Div([
                html.Div([
                    html.Div("Smart Mode", style={"fontWeight": "700", "marginBottom": ".3rem"}),
                    html.P("Fully automated protocol with up to three 5-minute stimulation blocks and adaptive rest periods. "
                           "Baseline and recovery EMG measurements determine break duration and session "
                           "stop criteria. Intended for the complete protocol workflow.", style={"margin": 0}),
                ], style={"flex": "1", "minWidth": "220px"}),
                html.Div([
                    html.Div("Manual Mode", style={"fontWeight": "700", "marginBottom": ".3rem"}),
                    html.P("Direct relay control and manually triggered measurements. Use for device "
                           "testing, electrode placement verification, and supervised single-session "
                           "operation. Clearly labelled as debug/supervised.", style={"margin": 0}),
                ], style={"flex": "1", "minWidth": "220px"}),
            ], className="body-text", style={"display": "flex", "gap": "1.5rem", "flexWrap": "wrap"}),
        ], className="card"),
    ])


def _tab_safety() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("Safety Information", className="section-title"),
            html.Div("AdaptStim is a research prototype. Read all cautions before use.", className="body-text"),
        ], className="card"),
        html.Div([
            html.Div("Absolute Contraindications", className="section-title"),
            html.Div([
                html.Ul([
                    html.Li("Implanted electronic devices (pacemakers, defibrillators, neurostimulators, cochlear implants) — do not use."),
                    html.Li("Pregnancy — do not use."),
                    html.Li("Active epilepsy or seizure disorder — consult a physician before use."),
                    html.Li("Known cardiac arrhythmia or serious heart disease — consult a physician."),
                    html.Li("Active cancer or major systemic illness — consult a physician."),
                ]),
            ], className="body-text danger-box"),
        ], className="card"),
        html.Div([
            html.Div("Warnings and Precautions", className="section-title"),
            html.Div([
                html.Ul([
                    html.Li("Do not use near the head, neck, or chest."),
                    html.Li("Do not use while driving, operating machinery, or in or near water."),
                    html.Li("Do not use on irritated, broken, or infected skin at the site of electrode placement."),
                    html.Li("Do not use in the presence of flammable gases or oxygen-enriched environments."),
                    html.Li("Keep away from children. This is a research device and is not intended for unsupervised use."),
                    html.Li("If you experience pain, burning, or unusual sensations, stop immediately and consult a physician."),
                ]),
            ], className="body-text warn-box"),
        ], className="card"),
    ])


def _tab_setup() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("Electrode Placement", className="section-title"),
            html.Div([
                html.P([html.Strong("Goal: "), "Make the ankle lift upward, called dorsiflexion. "
                        "A small outward turn of the foot, called eversion, is okay. Avoid strong toe lifting, "
                        "foot pointing downward, pain, or major discomfort."]),
                html.P([html.Strong("Main landmark: "), "Find the fibular head, the small bony bump on the outside "
                        "of the knee. The common peroneal nerve passes close to the skin around this area, so it is "
                        "a good place for the upper stimulation electrode."]),
                html.P([html.Strong("Proximal electrode: "), "Place the upper electrode near the fibular head/neck "
                        "area on the outside of the knee. It should be slightly below and slightly toward the "
                        "front/outside of the fibular head. Do not place it directly on the bony bump if it feels "
                        "uncomfortable."]),
                html.P([html.Strong("Distal electrode: "), "Place the lower electrode on the front-outside part of "
                        "the shin, over the tibialis anterior muscle. This is the muscle that helps lift the foot "
                        "upward. The electrode should be just outside the shin bone and slightly below the upper "
                        "electrode. Move it in small steps until stimulation causes good ankle dorsiflexion without "
                        "too much toe extension, eversion, or inversion."]),
                html.P(html.Strong("Adjustment guide:")),
                html.Ul([
                    html.Li("If the foot turns outward too much, move the lower electrode slightly more toward the front shin muscle and away from the outer lower-leg muscles."),
                    html.Li("If the foot turns inward too much, move the electrode slightly outward or lower the stimulation intensity."),
                    html.Li("If the toes lift more than the ankle, move the electrode away from the toe extensor area and back toward the tibialis anterior."),
                    html.Li("If the foot points downward, the electrode may be too far back and may be stimulating calf muscles or the tibial nerve."),
                ]),
                html.P([html.Strong("Skin preparation: "), "Clean the skin with an alcohol wipe and let it dry fully. "
                        "Remove extra hair if needed. Press the electrode edges down firmly so the full pad contacts "
                        "the skin."]),
                html.P([html.Strong("Storage: "), "Replace gel pads when they stop sticking well or when stimulation "
                        "feels weaker. Store unused pads sealed at room temperature."]),
            ], className="body-text"),
        ], className="card"),
        html.Div([
            html.Div("Port Connection", className="section-title"),
            html.Div([
                html.Ul([
                    html.Li("BYB EMG device: connect to COM4 (default) at 222,222 baud."),
                    html.Li("ESP32-S3 relay: connect to COM5 (default) at 230,400 baud."),
                    html.Li("Verify both ports are listed in Device Manager as 'USB Serial Device' with status OK."),
                    html.Li("The live signal graph appears in Smart Mode and Manual Mode when the BYB is streaming."),
                ]),
            ], className="body-text"),
        ], className="card"),
        html.Div([
            html.Div("TENS 7000 Unit Settings", className="section-title"),
            html.Div([
                html.P("Set the TENS unit to a comfortable pulse width (μs) and frequency (Hz) "
                       "before starting Smart Mode. The ESP32 relay enables or disables "
                       "the TENS output — it does not adjust TENS parameters during a session."),
            ], className="body-text"),
        ], className="card"),
    ])


def _tab_science() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("Background", className="section-title"),
            html.Div([
                html.P("Peroneal nerve stimulation excites the muscles of the anterior compartment of "
                       "the lower leg, producing rhythmic dorsiflexion and activating the calf muscle pump. "
                       "This promotes venous return and may reduce lower-limb discomfort during prolonged "
                       "sitting or immobility."),
                html.P("Adaptive rest cycles matter because repeated electrical stimulation can cause "
                       "local muscle fatigue. AdaptStim monitors RMS amplitude and Median Frequency (MDF) "
                       "during dedicated recovery measurement windows. MDF shifts downward as fatigue "
                       "accumulates, and RMS may increase with compensatory co-activation. These signals "
                       "guide break-length decisions."),
                html.P("RMS/MDF are computed only during baseline and recovery measurement windows — "
                       "not continuously during the live EMG trace. This avoids artefact-driven "
                       "decisions during stimulation."),
            ], className="body-text"),
        ], className="card"),
        html.Div([
            html.Div("Reference Placeholders", className="section-title"),
            html.Div([
                html.Ul([
                    html.Li([html.Span("Tucker et al., 2010 — "), html.Span("Neuromuscular electrical stimulation and venous haemodynamics.", className="ref-tag")]),
                    html.Li([html.Span("Izumi et al., 2015 — "), html.Span("Peroneal nerve stimulation and lower-limb blood flow.", className="ref-tag")]),
                    html.Li([html.Span("Grimm & Charles, 2018 — "), html.Span("Adaptive stimulation parameters and fatigue detection.", className="ref-tag")]),
                    html.Li([html.Span("Natarajan et al., 2021 — "), html.Span("Surface EMG-based fatigue monitoring during FES.", className="ref-tag")]),
                ]),
            ], className="body-text"),
        ], className="card"),
    ])


def _tab_troubleshoot() -> html.Div:
    items = [
        ("No stimulation output",
         "Check that the ESP32 relay is connected on COM5. Use Manual Mode -> Stim ON to test. "
         "Verify the TENS unit is powered on and set to the desired output level."),
        ("Weak or inconsistent stimulation",
         "Confirm electrode contact at the site of electrode placement. Clean the skin surface and "
         "reapply electrodes. Increase TENS intensity gradually. Check relay connections."),
        ("EMG signal not detected",
         "Verify BYB device is connected on COM4 at 222,222 baud. Check the top status dot and live signal graph. "
         "Ensure skin is clean and electrode contact is firm. Re-seat the BYB USB cable."),
        ("Port connection issue",
         "Open Device Manager and confirm both COM4 and COM5 appear as 'USB Serial Device (COMx)' with status OK. "
         "Try unplugging and reconnecting the USB cables. Restart the app if ports were connected after launch."),
        ("Electrode contact issue",
         "Replace electrode gel pads if they have dried out or lost adhesion. Clean the site of electrode placement "
         "with an alcohol wipe. Ensure the fibular head landmark is correctly identified before placing electrodes."),
        ("Recovery measurement failed quality check",
         "The signal quality gate requires at least 2 valid contraction windows with RMS above the noise floor and "
         "MDF within the physiological EMG band. Check electrode contact, confirm the subject is actively contracting "
         "during the CONTRACT cue, and retry the measurement."),
    ]
    return html.Div([
        html.Div([
            html.Div(title, className="section-title", style={"marginBottom": ".35rem"}),
            html.P(body, className="body-text", style={"marginBottom": 0}),
        ], className="card") for title, body in items
    ])


def _children_list(component) -> list:
    children = getattr(component, "children", [])
    return children if isinstance(children, list) else [children]


def _tab_overview() -> html.Div:
    return html.Div(_children_list(_tab_home()) + _children_list(_tab_science()))


def _tab_instructions() -> html.Div:
    return html.Div(
        _children_list(_tab_setup())
        + [html.Div("Troubleshoot", className="section-title", style={"margin": "1rem 0 .5rem"})]
        + _children_list(_tab_troubleshoot())
        + [html.Div("Safety", className="section-title", style={"margin": "1rem 0 .5rem"})]
        + _children_list(_tab_safety())
    )


def create_app(backend: AdaptStimBackend) -> dash.Dash:
    app = dash.Dash(__name__, title="AdaptStim", update_title=None,
                    suppress_callback_exceptions=True)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app_started_ms = int(time.time() * 1000)

    app.index_string = f"""<!DOCTYPE html>
<html>
<head>{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
<style>{_CSS}</style>
</head>
<body class="shell">
{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""

    app.layout = html.Div([
        dcc.Interval(id="tick", interval=200, n_intervals=0),
        dcc.Store(id="store-estop-ts", data=0),
        dcc.Store(id="store-fullstart-ts", data=0),
        dcc.Store(id="store-fullstop-ts", data=0),
        dcc.Store(id="store-demostart-ts", data=0),
        dcc.Store(id="store-demostop-ts", data=0),
        dcc.Store(id="store-demoreset-ts", data=0),
        dcc.Store(id="store-manual-enter-ts", data=0),
        dcc.Store(id="store-manual-exit-ts", data=0),
        dcc.Store(id="store-meas-start-ts", data=0),
        dcc.Store(id="store-reset-ts", data=0),

        # Top bar
        html.Div([
            html.Div([
                html.Div([html.Span("Adapt"), html.Span("Stim", style={"color": "#0891b2"})],
                         className="logo-text"),
                html.Div("Strap. Tap. Recover.", className="tagline"),
            ]),
            html.Div([
                html.Span(id="byb-dot", className="dot dot-err"),
                html.Span(id="byb-status-txt", style={"fontSize": ".8rem", "color": "#64748b"}),
            ], className="topbar-status"),
            html.Button("⏹ Emergency Stop", id="btn-estop", n_clicks=0, className="estop-btn"),
        ], className="topbar"),

        # Tabs
        dcc.Tabs(id="tabs", value="tab-full", children=[
            dcc.Tab(label="Smart Mode", value="tab-full"),
            dcc.Tab(label="Manual Mode", value="tab-manual"),
            dcc.Tab(label="Demo", value="tab-demo"),
            dcc.Tab(label="Overview", value="tab-home"),
            dcc.Tab(label="Instructions", value="tab-instructions"),
        ], style={"fontFamily": "Inter, sans-serif", "borderBottom": "1px solid #e2e8f0"}),

        html.Div(id="tab-content", className="view-content"),
    ], className="shell")

    # ── Tab content renderer ─────────────────────────────────────────────────

    @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(tab):
        def pane(tab_id: str, children):
            return html.Div(children, style={"display": "block" if tab == tab_id else "none"})

        return [
            pane("tab-full", [
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Smart Mode", className="section-title"),
                            html.Div([
                                html.Button("Start Session", id="btn-full-start", n_clicks=0,
                                            className="btn btn-primary"),
                                html.Button("Stop Session", id="btn-full-stop", n_clicks=0,
                                            className="btn btn-danger"),
                                html.Button("Reset / Clear Error", id="btn-reset", n_clicks=0,
                                            className="btn"),
                            ], className="btn-row", style={"marginBottom": "1rem"}),
                            html.Div([
                                html.Div(id="full-phase-badge"),
                                html.Div(id="full-countdown", className="countdown",
                                         style={"marginTop": ".5rem"}),
                                html.Div(id="full-status-msg",
                                         style={"fontSize": ".88rem", "color": "#64748b", "marginTop": ".35rem"}),
                            ], className="card"),
                            html.Div([
                                html.Div("Session Info", className="card-title"),
                                html.Div([
                                    _metric("Cycle", "full-cycle"),
                                    _metric("Baseline RMS", "full-rms-base"),
                                    _metric("Baseline MDF", "full-mdf-base"),
                                    _metric("Last RMS", "full-rms-last"),
                                    _metric("Last MDF", "full-mdf-last"),
                                    _metric("Break Assigned", "full-break"),
                                    _metric("Measurements", "full-n-meas"),
                                ], className="metric-grid"),
                            ], className="card"),
                            html.Div(id="full-cue-area"),
                            html.Div(id="full-error-area"),
                            html.Div(id="full-log-area"),
                        ], className="control-stack"),
                        _live_signal_panel("full"),
                    ], className="work-grid"),
                ], className="content"),
            ]),

            pane("tab-demo", [
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Demo", className="section-title"),
                            html.Div([
                                html.Button("Start Demo", id="btn-demo-start", n_clicks=0,
                                            className="btn btn-primary"),
                                html.Button("Stop Demo", id="btn-demo-stop", n_clicks=0,
                                            className="btn btn-danger"),
                                html.Button("Reset / Clear Error", id="btn-demo-reset", n_clicks=0,
                                            className="btn"),
                            ], className="btn-row", style={"marginBottom": "1rem"}),
                            html.Div([
                                html.Div(id="demo-phase-badge"),
                                html.Div(id="demo-countdown", className="countdown",
                                         style={"marginTop": ".5rem"}),
                                html.Div(id="demo-status-msg",
                                         style={"fontSize": ".88rem", "color": "#64748b", "marginTop": ".35rem"}),
                            ], className="card"),
                            html.Div([
                                html.Div("Demo Info", className="card-title"),
                                html.Div([
                                    _metric("Cycle", "demo-cycle"),
                                    _metric("Baseline RMS", "demo-rms-base"),
                                    _metric("Baseline MDF", "demo-mdf-base"),
                                    _metric("Last RMS", "demo-rms-last"),
                                    _metric("Last MDF", "demo-mdf-last"),
                                    _metric("Break Assigned", "demo-break"),
                                    _metric("Measurements", "demo-n-meas"),
                                ], className="metric-grid"),
                            ], className="card"),
                            html.Div(id="demo-cue-area"),
                            html.Div(id="demo-error-area"),
                            html.Div(id="demo-log-area"),
                        ], className="control-stack"),
                        _live_signal_panel("demo"),
                    ], className="work-grid"),
                ], className="content"),
            ]),

            pane("tab-manual", [
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Manual Mode", className="section-title"),
                            html.Div("Manual / Debug / Supervised Operation", className="manual-label"),
                            html.P("This mode is for device testing and supervised single-session operation. "
                                   "Relay controls and RMS/MDF measurements are available here.",
                                   className="body-text", style={"marginBottom": ".75rem"}),
                            html.Div([
                                html.Button("Enter Manual Mode", id="btn-manual-enter", n_clicks=0,
                                            className="btn btn-primary"),
                                html.Button("Exit Manual Mode", id="btn-manual-exit", n_clicks=0,
                                            className="btn"),
                            ], className="btn-row", style={"marginBottom": "1rem"}),
                            html.Div([
                                html.Div("Relay Control", className="card-title"),
                                html.Div([
                                    html.Button("Stim ON", id="btn-stim-on", n_clicks=0,
                                                className="btn btn-primary"),
                                    html.Button("Stim OFF", id="btn-stim-off", n_clicks=0,
                                                className="btn btn-danger"),
                                    html.Button("Release Port", id="btn-esp32-release", n_clicks=0,
                                                className="btn"),
                                ], className="btn-row"),
                                html.Div(id="esp32-status-txt",
                                         style={"fontSize": ".8rem", "color": "#64748b", "marginTop": ".5rem"}),
                            ], className="card"),
                            html.Div([
                                html.Div("Manual Measurement (30 s contract/relax)", className="card-title"),
                                html.Button("Start Measurement", id="btn-meas-start", n_clicks=0,
                                            className="btn btn-primary", style={"marginBottom": ".75rem"}),
                                html.Div(id="manual-cue-area"),
                                html.Div(id="manual-result-area"),
                            ], className="card"),
                        ], className="control-stack"),
                        _live_signal_panel("smart"),
                    ], className="work-grid"),
                ], className="content"),
            ]),

            pane("tab-home", html.Div(_tab_overview(), className="content")),
            pane("tab-instructions", html.Div(_tab_instructions(), className="content")),
        ]

    # ── Button → Store callbacks ─────────────────────────────────────────────

    @app.callback(Output("store-estop-ts", "data"),
                  Input("btn-estop", "n_clicks"),
                  State("btn-estop", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_estop(_, ts):
        if ts and ts >= app_started_ms:
            backend.emergency_stop()
        return ts or 0

    @app.callback(Output("store-fullstart-ts", "data"),
                  Input("btn-full-start", "n_clicks"),
                  State("btn-full-start", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_full_start(_, ts):
        if ts and ts >= app_started_ms:
            backend.start_full_device()
        return ts or 0

    @app.callback(Output("store-fullstop-ts", "data"),
                  Input("btn-full-stop", "n_clicks"),
                  State("btn-full-stop", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_full_stop(_, ts):
        if ts and ts >= app_started_ms:
            backend.stop_full_device()
        return ts or 0

    @app.callback(Output("store-demostart-ts", "data"),
                  Input("btn-demo-start", "n_clicks"),
                  State("btn-demo-start", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_demo_start(_, ts):
        if ts and ts >= app_started_ms:
            backend.start_demo()
        return ts or 0

    @app.callback(Output("store-demostop-ts", "data"),
                  Input("btn-demo-stop", "n_clicks"),
                  State("btn-demo-stop", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_demo_stop(_, ts):
        if ts and ts >= app_started_ms:
            backend.stop_full_device()
        return ts or 0

    @app.callback(Output("store-demoreset-ts", "data"),
                  Input("btn-demo-reset", "n_clicks"),
                  State("btn-demo-reset", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_demo_reset(_, ts):
        if ts and ts >= app_started_ms:
            backend.reset_from_error()
        return ts or 0

    @app.callback(Output("store-reset-ts", "data"),
                  Input("btn-reset", "n_clicks"),
                  State("btn-reset", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_reset(_, ts):
        if ts and ts >= app_started_ms:
            backend.reset_from_error()
        return ts or 0

    @app.callback(Output("store-manual-enter-ts", "data"),
                  Input("btn-manual-enter", "n_clicks"),
                  State("btn-manual-enter", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_manual_enter(_, ts):
        if ts and ts >= app_started_ms:
            backend.enter_manual()
        return ts or 0

    @app.callback(Output("store-manual-exit-ts", "data"),
                  Input("btn-manual-exit", "n_clicks"),
                  State("btn-manual-exit", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_manual_exit(_, ts):
        if ts and ts >= app_started_ms:
            backend.exit_manual()
        return ts or 0

    @app.callback(Output("store-meas-start-ts", "data"),
                  Input("btn-meas-start", "n_clicks"),
                  State("btn-meas-start", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def on_meas_start(_, ts):
        if ts and ts >= app_started_ms:
            backend.start_manual_measurement()
        return ts or 0

    # ── ESP32 manual controls ────────────────────────────────────────────────

    @app.callback(Output("esp32-status-txt", "children"),
                  Input("btn-stim-on", "n_clicks"),
                  Input("btn-stim-off", "n_clicks"),
                  Input("btn-esp32-release", "n_clicks"),
                  Input("tick", "n_intervals"),
                  State("btn-stim-on", "n_clicks_timestamp"),
                  State("btn-stim-off", "n_clicks_timestamp"),
                  State("btn-esp32-release", "n_clicks_timestamp"),
                  prevent_initial_call=True)
    def update_esp32(_, __, ___, _t, ts_on, ts_off, ts_rel):
        tid = callback_context.triggered[0]["prop_id"].split(".")[0]
        if tid == "btn-stim-on" and ts_on and ts_on >= app_started_ms:
            threading.Thread(target=backend.esp32.stim_on, daemon=True).start()
        elif tid == "btn-stim-off" and ts_off and ts_off >= app_started_ms:
            threading.Thread(target=backend.esp32.stim_off, daemon=True).start()
        elif tid == "btn-esp32-release" and ts_rel and ts_rel >= app_started_ms:
            backend.esp32.release()
        s = backend.esp32.snapshot()
        parts = []
        if s["last_time"]:
            parts.append(s["last_time"])
        parts.append("connected" if s["connected"] else "disconnected")
        parts.append(s["last_msg"])
        return " | ".join(parts)

    # ── Main live update ─────────────────────────────────────────────────────

    @app.callback(
        Output("byb-dot", "className"),
        Output("byb-status-txt", "children"),
        Output("full-phase-badge", "children"),
        Output("full-countdown", "children"),
        Output("full-status-msg", "children"),
        Output("full-cycle", "children"),
        Output("full-rms-base", "children"),
        Output("full-mdf-base", "children"),
        Output("full-rms-last", "children"),
        Output("full-mdf-last", "children"),
        Output("full-break", "children"),
        Output("full-n-meas", "children"),
        Output("full-cue-area", "children"),
        Output("full-error-area", "children"),
        Output("full-log-area", "children"),
        Output("demo-phase-badge", "children"),
        Output("demo-countdown", "children"),
        Output("demo-status-msg", "children"),
        Output("demo-cycle", "children"),
        Output("demo-rms-base", "children"),
        Output("demo-mdf-base", "children"),
        Output("demo-rms-last", "children"),
        Output("demo-mdf-last", "children"),
        Output("demo-break", "children"),
        Output("demo-n-meas", "children"),
        Output("demo-cue-area", "children"),
        Output("demo-error-area", "children"),
        Output("demo-log-area", "children"),
        Output("manual-cue-area", "children"),
        Output("manual-result-area", "children"),
        Input("tick", "n_intervals"),
    )
    def live_update(_):
        s = backend.snapshot(window_seconds=10.0)
        state = s["state"]

        # BYB status dot
        age = s["sample_age"]
        if s["connected"] and age is not None and age < 1.5:
            dot_cls = "dot dot-ok"
            byb_txt = f"Streaming · {s['sample_rate']:.0f} Hz"
        elif s["connected"]:
            dot_cls = "dot dot-warn"
            byb_txt = "Connected — no recent samples"
        else:
            dot_cls = "dot dot-err"
            byb_txt = s["byb_msg"]

        # Phase badge
        color = _phase_color(state)
        badge = html.Span(_phase_label(state), className="phase-badge",
                          style={"background": color})

        # Countdown
        countdown = _fmt_countdown(s["phase_remaining"]) if s["phase_remaining"] is not None else ""

        # Cue area (Smart Mode measurement phases)
        cue_area = []
        if s["meas_cue"] and state in (AppState.INITIAL_BASELINE, AppState.RECOVERY_MEASUREMENT):
            cue = s["meas_cue"]
            c = cue["cue"]
            cue_color = _cue_color(c)
            cue_area = [html.Div([
                html.Div("Measurement Cue", className="card-title"),
                html.Div(c, className="cue-box", style={"background": cue_color}),
                html.Div([
                    html.Div(style={"width": f"{min(cue['elapsed'] / cue['duration'] * 100, 100):.1f}%",
                                    "background": cue_color},
                             className="progress-fill"),
                ], className="progress-track"),
                html.Div(f"{_fmt_countdown(cue['remaining'])} remaining in window",
                         style={"fontSize": ".8rem", "color": "#64748b", "marginTop": ".35rem"}),
            ], className="card")]

        # Error area
        error_area = []
        if s["error_msg"]:
            error_area = [html.Div([
                html.Div(s["error_msg"], style={"fontWeight": "600", "color": "#dc2626"}),
                html.P("Check electrode contact, ensure BYB is streaming, then click Reset / Clear Error.",
                       className="body-text", style={"marginTop": ".35rem", "marginBottom": 0}),
            ], className="card danger-box")]
        if s["stop_reason"] and state == AppState.COMPLETE:
            error_area = [html.Div([
                html.Div("Session Complete", style={"fontWeight": "700", "marginBottom": ".3rem"}),
                html.Div(s["stop_reason"], style={"fontSize": ".88rem", "color": "#334155"}),
            ], className="card", style={"borderColor": "#059669"})]

        # Manual cue area
        man_cue_area = []
        if s["man_cue"]:
            cue = s["man_cue"]
            c = cue["cue"]
            cc = _cue_color(c)
            man_cue_area = [
                html.Div(c, className="cue-box", style={"background": cc, "marginBottom": ".5rem"}),
                html.Div([
                    html.Div(style={"width": f"{min(cue['elapsed'] / cue['duration'] * 100, 100):.1f}%",
                                    "background": cc},
                             className="progress-fill"),
                ], className="progress-track"),
                html.Div(f"{_fmt_countdown(cue['remaining'])} remaining",
                         style={"fontSize": ".8rem", "color": "#64748b", "marginTop": ".35rem"}),
            ]

        # Manual result area
        man_result_area = []
        mr = s["manual_result"]
        if mr:
            if mr["quality_ok"]:
                man_result_area = [html.Div([
                    html.Div("Measurement Result", className="card-title"),
                    html.Div([
                        html.Div([html.Div("RMS", className="metric-label"),
                                  html.Div(_fmt(mr["rms"]), className="metric-value")], className="metric-card"),
                        html.Div([html.Div("MDF", className="metric-label"),
                                  html.Div(_fmt(mr["mdf"], suffix=" Hz"), className="metric-value")], className="metric-card"),
                        html.Div([html.Div("Windows", className="metric-label"),
                                  html.Div(str(mr["n_windows"]), className="metric-value")], className="metric-card"),
                        html.Div([html.Div("Time", className="metric-label"),
                                  html.Div(mr["timestamp"], className="metric-value")], className="metric-card"),
                    ], className="metric-grid"),
                ], className="card")]
            else:
                man_result_area = [html.Div([
                    html.Div("Quality check failed: " + mr["quality_reason"],
                             style={"fontSize": ".88rem", "color": "#dc2626", "fontWeight": "600"}),
                ], className="card danger-box")]

        # Smart Mode session info
        lr = s["last_result"]
        break_text = f"{int(s['break_seconds'] // 60)} min" if s.get("break_seconds") else "—"
        log_area = _readings_table(s.get("measurements_log", []))
        last_rms = _fmt(lr["rms"]) if lr and lr.get("quality_ok") else "—"
        last_mdf = _fmt(lr["mdf"], suffix=" Hz") if lr and lr.get("quality_ok") else "—"
        return (
            dot_cls, byb_txt,
            badge, countdown, s["status_msg"],
            str(s["cycle"]),
            _fmt(s["first_baseline_rms"]),
            _fmt(s["first_baseline_mdf"], suffix=" Hz"),
            last_rms,
            last_mdf,
            break_text,
            str(s["n_measurements"]),
            cue_area,
            error_area,
            log_area,
            badge,
            countdown,
            s["status_msg"],
            str(s["cycle"]),
            _fmt(s["first_baseline_rms"]),
            _fmt(s["first_baseline_mdf"], suffix=" Hz"),
            last_rms,
            last_mdf,
            break_text,
            str(s["n_measurements"]),
            cue_area,
            error_area,
            log_area,
            man_cue_area,
            man_result_area,
        )

    # ── Live signal graph ────────────────────────────────────────────────────

    @app.callback(
        Output("signal-graph-full", "figure"),
        Output("signal-graph-smart", "figure"),
        Output("signal-graph-demo", "figure"),
        Input("tick", "n_intervals"),
        Input("tabs", "value"),
        State("window-seconds-full", "value"),
        State("window-seconds-smart", "value"),
        State("window-seconds-demo", "value"),
        prevent_initial_call=False,
    )
    def update_graph(_, tab, full_window_seconds, smart_window_seconds, demo_window_seconds):
        if tab == "tab-manual":
            w = float(smart_window_seconds or 10)
            s = backend.snapshot(window_seconds=w)
            return dash.no_update, build_figure(s["x"], s["y"], w), dash.no_update
        if tab == "tab-full":
            w = float(full_window_seconds or 10)
            s = backend.snapshot(window_seconds=w)
            return build_figure(s["x"], s["y"], w), dash.no_update, dash.no_update
        if tab == "tab-demo":
            w = float(demo_window_seconds or 10)
            s = backend.snapshot(window_seconds=w)
            return dash.no_update, dash.no_update, build_figure(s["x"], s["y"], w)
        return dash.no_update, dash.no_update, dash.no_update

    @app.server.get("/health")
    def health():
        s = backend.snapshot()
        return {"state": s["state"], "connected": s["connected"],
                "sample_rate": s["sample_rate"], "esp32": backend.esp32.snapshot()}

    return app


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def find_free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port near {preferred}")


def parse_args():
    p = argparse.ArgumentParser(description="AdaptStim adaptive peroneal stimulation prototype")
    p.add_argument("--port", default=DEFAULT_BYB_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--esp-port", default=DEFAULT_ESP32_PORT)
    p.add_argument("--esp-baud", type=int, default=DEFAULT_ESP32_BAUD)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    web_port = find_free_port(args.web_port)
    url = f"http://{args.host}:{web_port}"

    esp32 = ESP32Controller(args.esp_port, args.esp_baud)
    backend = AdaptStimBackend(args.port, args.baud, esp32)
    backend.start()
    app = create_app(backend)

    print(f"BYB input: {args.port} at {args.baud} baud")
    print(f"ESP32:     {args.esp_port} at {args.esp_baud} baud")
    print(f"Viewer:    {url}")
    print("Close this terminal or press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(0.8), webbrowser.open(url)),
            daemon=True
        ).start()

    try:
        try:
            app.run(host=args.host, port=web_port, debug=False, threaded=True)
        except AttributeError:
            app.run_server(host=args.host, port=web_port, debug=False, threaded=True)
    finally:
        backend.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
