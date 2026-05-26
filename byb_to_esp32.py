"""
Backyard Brains EMG to ESP32-S3 stimulation relay bridge.

Session protocol:
1. Start with relay OFF.
2. Run a 1-minute guided EMG assessment to establish the pre-stimulation
   reference values.
3. Turn relay ON for a 15-minute stimulation cycle.
4. Turn relay OFF.
5. Run another 1-minute guided EMG assessment.
6. Continue another 15-minute cycle only if flex median frequency (MDF) is
   still below 90% of the initial flex MDF reference.

Before running:
- Close Spike Recorder so it is not holding COM4.
- Close Arduino Serial Monitor / Serial Plotter so they are not holding COM3.
- Install packages with:
    python -m pip install pyserial numpy scipy
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import time

try:
    import numpy as np
    from scipy import signal
except ImportError:
    print("ERROR: numpy and scipy are required for EMG RMS/MDF analysis.")
    print("Install them with:")
    print("  python -m pip install numpy scipy")
    sys.exit(1)

try:
    import serial
    from serial import SerialException
except ImportError:
    print("ERROR: pyserial is not installed.")
    print("Install it with:")
    print("  python -m pip install pyserial")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Easy-to-tune serial settings
# ---------------------------------------------------------------------------

BYB_PORT = "COM4"
ESP32_PORT = "COM3"

BYB_BAUD = 222222
ESP32_BAUD = 115200

# Your current hardware behaves inverted from the original assumption:
# sending serial "OFF" physically turns stimulation ON, and sending serial
# "ON" physically turns stimulation OFF.
#
# The rest of the script uses logical relay states:
# - logical "ON" means stimulation should be active
# - logical "OFF" means stimulation should be inactive / safe
#
# If you later fix the ESP32 sketch or relay wiring so ON really is ON,
# change these two lines to:
# RELAY_ON_SERIAL_COMMAND = "ON"
# RELAY_OFF_SERIAL_COMMAND = "OFF"
RELAY_ON_SERIAL_COMMAND = "OFF"
RELAY_OFF_SERIAL_COMMAND = "ON"


# ---------------------------------------------------------------------------
# Easy-to-tune stimulation and assessment settings
# ---------------------------------------------------------------------------

STIM_DURATION_SECONDS = 15 * 60
ASSESSMENT_DURATION_SECONDS = 60
RESTING_WINDOW_SECONDS = 20
FLEX_DURATION_SECONDS = 5
RELAX_BETWEEN_FLEXES_SECONDS = 8
MDF_RECOVERY_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Easy-to-tune EMG analysis settings
# ---------------------------------------------------------------------------

MIN_ANALYSIS_SAMPLES = 100
EMG_HIGHPASS_HZ = 20.0
EMG_LOWPASS_HZ = 450.0
WELCH_WINDOW_SECONDS = 1.0


@dataclass
class EMGWindow:
    """Samples and estimated sampling rate collected during one time window."""

    samples: list
    sample_rate_hz: float | None
    duration_seconds: float


@dataclass
class AssessmentMetrics:
    """Summary values from one guided assessment block."""

    resting_rms: float
    average_flex_rms: float
    average_flex_mdf: float
    flex_rms_values: list
    flex_mdf_values: list


class BYBSampleReader:
    """
    Read Backyard Brains samples using the common 2-byte framing format.

    Expected byte format:
    - First byte: high bit set, carries the upper 7 sample bits.
    - Second byte: high bit clear, carries the lower 7 sample bits.

    Decoding:
    - Remove the high bit from the first byte.
    - Reconstruct the sample as (msb << 7) | lsb.
    - Center around zero by subtracting 512.

    The small state machine below lets the reader resynchronize if we start
    reading in the middle of the serial stream.
    """

    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.msb = None

    def clear_buffer(self):
        """Discard old bytes so the next collection window uses fresh EMG."""
        self.msb = None
        try:
            self.serial_port.reset_input_buffer()
        except SerialException:
            pass

    def read_sample(self, max_bytes_to_scan=4096):
        """
        Return one centered sample, or None if no complete sample is available.

        The serial port timeout controls how long this method waits for bytes.
        The byte scan limit prevents an endless loop if the BYB device is using
        a different packet format than the one this script expects.
        """
        bytes_scanned = 0

        while bytes_scanned < max_bytes_to_scan:
            byte_data = self.serial_port.read(1)

            if not byte_data:
                return None

            bytes_scanned += 1
            byte_value = byte_data[0]

            if byte_value & 0x80:
                # This is a first byte. Store its lower 7 bits as the MSB.
                self.msb = byte_value & 0x7F
                continue

            if self.msb is None:
                # We saw a second byte before seeing a first byte. Ignore it.
                continue

            # This is a valid second byte. It carries the lower 7 bits.
            lsb = byte_value & 0x7F
            raw_sample = (self.msb << 7) | lsb
            self.msb = None

            centered_sample = raw_sample - 512
            return centered_sample

        return None


def open_serial_port(port_name, baud_rate, label):
    """
    Open a serial port and print a useful beginner-friendly error if it fails.
    """
    try:
        serial_port = serial.Serial(
            port=port_name,
            baudrate=baud_rate,
            timeout=0.1,
        )
    except SerialException as error:
        print(f"\nERROR: Could not open {label} serial port {port_name}.")
        print(f"Baud rate requested: {baud_rate}")
        print(f"Serial error: {error}")
        print("\nThings to check:")
        if port_name == BYB_PORT:
            print("- Close Spike Recorder if it is open.")
            print("- Confirm the Backyard Brains device is really on COM4.")
        if port_name == ESP32_PORT:
            print("- Close Arduino Serial Monitor / Serial Plotter if open.")
            print("- Confirm the ESP32-S3 is really on COM3.")
        print("- Unplug/replug the USB cable and check Device Manager if needed.")
        return None

    print(f"Opened {label} on {port_name} at {baud_rate} baud.")
    return serial_port


def send_relay_command(esp32_serial, command):
    """
    Send a logical relay command to the ESP32-S3 and print the mapping.

    The ESP32 expects exactly "ON\n" or "OFF\n".
    This function maps logical stimulation ON/OFF to the serial text that
    physically produces that state on the current hardware.
    """
    if command not in ("ON", "OFF"):
        raise ValueError("Relay command must be 'ON' or 'OFF'.")

    if command == "ON":
        serial_command = RELAY_ON_SERIAL_COMMAND
    else:
        serial_command = RELAY_OFF_SERIAL_COMMAND

    esp32_serial.write(f"{serial_command}\n".encode("ascii"))
    esp32_serial.flush()
    print(f"Logical relay {command}: sent serial '{serial_command}'")


def safe_send_off(esp32_serial):
    """Best-effort fail-safe OFF command used during errors and shutdown."""
    if esp32_serial is None:
        return

    try:
        send_relay_command(esp32_serial, "OFF")
    except Exception as error:
        print(f"Could not send fail-safe OFF command: {error}")


def print_byb_data_help():
    """Explain likely causes when no useful Backyard Brains data is detected."""
    print("\nLikely BYB data reasons:")
    print("- Spike Recorder is still open and holding COM4.")
    print("- COM4 is not the correct port for the Backyard Brains device.")
    print(
        "- This BYB model may use a different baud rate, such as 500000 "
        "for some Human-Human Interface units."
    )
    print("- This BYB model may use a different packet format.")


def format_seconds(seconds):
    """Return a compact mm:ss countdown string."""
    seconds = max(0, int(round(seconds)))
    minutes = seconds // 60
    remainder = seconds % 60
    return f"{minutes:02d}:{remainder:02d}"


def collect_emg_for_duration(sample_reader, duration_seconds, instruction):
    """
    Collect centered EMG samples for a fixed duration.

    The function also estimates sample rate from the first and last received
    sample. MDF needs a sample rate so the spectrum can be labeled in Hz.
    """
    sample_reader.clear_buffer()

    samples = []
    first_sample_time = None
    last_sample_time = None
    start_time = time.monotonic()
    last_status_time = 0.0

    print(instruction)

    while True:
        now = time.monotonic()
        elapsed = now - start_time

        if elapsed >= duration_seconds:
            break

        sample = sample_reader.read_sample()
        now = time.monotonic()

        if sample is not None:
            samples.append(sample)
            if first_sample_time is None:
                first_sample_time = now
            last_sample_time = now

        if now - last_status_time >= 1.0:
            remaining = duration_seconds - (now - start_time)
            print(
                f"\r{instruction} "
                f"Remaining {format_seconds(remaining)} | "
                f"samples {len(samples)}",
                end="",
                flush=True,
            )
            last_status_time = now

    print()

    sample_rate_hz = None
    if (
        first_sample_time is not None
        and last_sample_time is not None
        and last_sample_time > first_sample_time
        and len(samples) >= 2
    ):
        sample_rate_hz = (len(samples) - 1) / (last_sample_time - first_sample_time)

    return EMGWindow(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )


def drain_emg_for_duration(sample_reader, duration_seconds, instruction):
    """
    Keep reading BYB bytes during relax/buffer periods without saving samples.

    This prevents old serial bytes from piling up and being analyzed in the
    next flex window.
    """
    sample_reader.clear_buffer()

    start_time = time.monotonic()
    last_status_time = 0.0

    print(instruction)

    while True:
        now = time.monotonic()
        elapsed = now - start_time

        if elapsed >= duration_seconds:
            break

        sample_reader.read_sample(max_bytes_to_scan=512)
        now = time.monotonic()

        if now - last_status_time >= 1.0:
            remaining = duration_seconds - (now - start_time)
            print(
                f"\r{instruction} Remaining {format_seconds(remaining)}",
                end="",
                flush=True,
            )
            last_status_time = now

    print()


def validate_emg_window(emg_window, label):
    """Raise a clear error if a collected EMG window cannot be analyzed."""
    if len(emg_window.samples) < MIN_ANALYSIS_SAMPLES:
        raise RuntimeError(
            f"Not enough BYB samples during {label}. "
            f"Got {len(emg_window.samples)}, need at least {MIN_ANALYSIS_SAMPLES}."
        )

    if emg_window.sample_rate_hz is None or emg_window.sample_rate_hz <= 0:
        raise RuntimeError(f"Could not estimate sample rate during {label}.")


def preprocess_emg(samples, sample_rate_hz):
    """
    Convert raw centered samples into filtered EMG for RMS/MDF analysis.

    Filtering approach:
    - Remove DC by subtracting the mean.
    - Apply a 20-450 Hz bandpass when the measured sample rate supports it.
    - If the sample rate is too low for the full bandpass, use a highpass when
      possible, otherwise fall back to detrended data.
    """
    data = np.asarray(samples, dtype=float)
    data = data - np.mean(data)

    if len(data) < MIN_ANALYSIS_SAMPLES or sample_rate_hz <= 0:
        return data

    nyquist_hz = sample_rate_hz / 2.0
    highpass_hz = EMG_HIGHPASS_HZ
    lowpass_hz = min(EMG_LOWPASS_HZ, nyquist_hz * 0.90)

    sos = None

    try:
        if lowpass_hz > highpass_hz:
            sos = signal.butter(
                4,
                [highpass_hz / nyquist_hz, lowpass_hz / nyquist_hz],
                btype="bandpass",
                output="sos",
            )
        elif highpass_hz < nyquist_hz * 0.90:
            sos = signal.butter(
                4,
                highpass_hz / nyquist_hz,
                btype="highpass",
                output="sos",
            )
        else:
            return data

        return signal.sosfiltfilt(sos, data)
    except ValueError:
        # Very short windows can fail the zero-phase filter padding requirement.
        # A one-pass filter is less ideal, but better than silently returning
        # unfiltered data when there are enough samples to analyze.
        if sos is None:
            return data
        return signal.sosfilt(sos, data)


def compute_rms(samples, sample_rate_hz):
    """Compute root-mean-square amplitude for one EMG window."""
    filtered = preprocess_emg(samples, sample_rate_hz)
    return float(np.sqrt(np.mean(filtered * filtered)))


def compute_median_frequency(samples, sample_rate_hz):
    """
    Compute median frequency from the EMG power spectrum.

    Median frequency is the frequency where cumulative spectral power reaches
    50% of total spectral power.
    """
    filtered = preprocess_emg(samples, sample_rate_hz)

    if len(filtered) < MIN_ANALYSIS_SAMPLES:
        raise RuntimeError("Not enough samples to compute MDF.")

    nperseg = min(len(filtered), max(64, int(sample_rate_hz * WELCH_WINDOW_SECONDS)))
    frequencies, power = signal.welch(
        filtered,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    # Ignore the DC bin. The EMG has already been detrended, and MDF should
    # reflect muscle signal power rather than offset.
    frequencies = frequencies[1:]
    power = power[1:]

    total_power = float(np.sum(power))
    if total_power <= 0 or not math.isfinite(total_power):
        raise RuntimeError("Power spectrum had no usable power for MDF.")

    cumulative_power = np.cumsum(power)
    halfway_power = total_power * 0.5
    median_index = int(np.searchsorted(cumulative_power, halfway_power))
    median_index = min(median_index, len(frequencies) - 1)

    return float(frequencies[median_index])


def percent_of_baseline(value, baseline_value):
    """Return value as percent of baseline, or None if baseline is zero."""
    if baseline_value == 0:
        return None
    return (value / baseline_value) * 100.0


def format_percent(percent_value):
    """Format a percentage that may be unavailable."""
    if percent_value is None:
        return "n/a"
    return f"{percent_value:.1f}%"


def print_assessment_summary(metrics, baseline_metrics=None):
    """Print the assessment metrics in a compact, readable block."""
    print("\nAssessment summary")
    print("------------------")
    print(f"Resting RMS:      {metrics.resting_rms:.2f}")
    print(f"Average flex RMS: {metrics.average_flex_rms:.2f}")
    print(f"Average flex MDF: {metrics.average_flex_mdf:.2f} Hz")

    if baseline_metrics is not None:
        resting_percent = percent_of_baseline(
            metrics.resting_rms,
            baseline_metrics.resting_rms,
        )
        flex_rms_percent = percent_of_baseline(
            metrics.average_flex_rms,
            baseline_metrics.average_flex_rms,
        )
        flex_mdf_percent = percent_of_baseline(
            metrics.average_flex_mdf,
            baseline_metrics.average_flex_mdf,
        )

        print()
        print(f"Resting RMS vs baseline:      {format_percent(resting_percent)}")
        print(f"Average flex RMS vs baseline: {format_percent(flex_rms_percent)}")
        print(f"Average flex MDF vs baseline: {format_percent(flex_mdf_percent)}")


def analyze_flex_window(emg_window, label):
    """Compute RMS and MDF for one standardized flex window."""
    validate_emg_window(emg_window, label)
    rms = compute_rms(emg_window.samples, emg_window.sample_rate_hz)
    mdf = compute_median_frequency(emg_window.samples, emg_window.sample_rate_hz)
    print(
        f"{label}: RMS {rms:.2f}, MDF {mdf:.2f} Hz, "
        f"estimated sample rate {emg_window.sample_rate_hz:.1f} Hz"
    )
    return rms, mdf


def run_assessment_block(sample_reader, phase_name, baseline_metrics=None):
    """
    Run the 1-minute guided assessment sequence.

    First 20 seconds:
    - Relax target muscle.
    - Compute resting RMS.

    Remaining 40 seconds:
    - Three standardized gentle flexes.
    - Compute RMS and MDF in each 5-second flex window.
    """
    print()
    print("=" * 72)
    print(phase_name)
    print("=" * 72)
    print("Relay should be OFF during assessment.")

    resting_window = collect_emg_for_duration(
        sample_reader,
        RESTING_WINDOW_SECONDS,
        "Relax target muscle...",
    )
    validate_emg_window(resting_window, "resting assessment")
    resting_rms = compute_rms(resting_window.samples, resting_window.sample_rate_hz)
    print(
        f"Resting window: RMS {resting_rms:.2f}, "
        f"estimated sample rate {resting_window.sample_rate_hz:.1f} Hz"
    )

    flex_rms_values = []
    flex_mdf_values = []

    for flex_number in range(1, 4):
        flex_window = collect_emg_for_duration(
            sample_reader,
            FLEX_DURATION_SECONDS,
            f"Flex gently now... ({flex_number}/3)",
        )
        rms, mdf = analyze_flex_window(
            flex_window,
            f"Flex {flex_number}",
        )
        flex_rms_values.append(rms)
        flex_mdf_values.append(mdf)

        if flex_number < 3:
            drain_emg_for_duration(
                sample_reader,
                RELAX_BETWEEN_FLEXES_SECONDS,
                "Relax...",
            )

    used_seconds = (
        RESTING_WINDOW_SECONDS
        + (3 * FLEX_DURATION_SECONDS)
        + (2 * RELAX_BETWEEN_FLEXES_SECONDS)
    )
    buffer_seconds = ASSESSMENT_DURATION_SECONDS - used_seconds

    if buffer_seconds > 0:
        drain_emg_for_duration(
            sample_reader,
            buffer_seconds,
            "Relax / buffer...",
        )

    metrics = AssessmentMetrics(
        resting_rms=resting_rms,
        average_flex_rms=float(np.mean(flex_rms_values)),
        average_flex_mdf=float(np.mean(flex_mdf_values)),
        flex_rms_values=flex_rms_values,
        flex_mdf_values=flex_mdf_values,
    )

    print_assessment_summary(metrics, baseline_metrics)
    return metrics


def run_stimulation_cycle(esp32_serial, sample_reader, cycle_number):
    """
    Turn stimulation ON continuously for one 15-minute cycle.

    BYB bytes are drained during stimulation so the later assessment does not
    accidentally analyze stale serial-buffer data.
    """
    print()
    print("=" * 72)
    print(f"Stimulation cycle {cycle_number}")
    print("=" * 72)

    sample_reader.clear_buffer()
    send_relay_command(esp32_serial, "ON")

    start_time = time.monotonic()
    last_status_time = 0.0

    while True:
        now = time.monotonic()
        elapsed = now - start_time

        if elapsed >= STIM_DURATION_SECONDS:
            break

        sample_reader.read_sample(max_bytes_to_scan=512)
        now = time.monotonic()

        if now - last_status_time >= 1.0:
            remaining = STIM_DURATION_SECONDS - (now - start_time)
            print(
                f"\rStimulating... cycle {cycle_number} | "
                f"remaining {format_seconds(remaining)}",
                end="",
                flush=True,
            )
            last_status_time = now

    print()
    send_relay_command(esp32_serial, "OFF")
    print(f"Stimulation cycle {cycle_number} complete.")


def should_continue_stimulation(post_metrics, baseline_metrics):
    """
    Decide whether another stimulation block is needed.

    First-version rule:
    - MDF is the actual decision variable.
    - Continue if post-cycle flex MDF is below 90% of initial flex MDF.

    Future optional improvement:
    - RMS could be added as a supporting rule, but it is effort-sensitive and
      is intentionally not used to control the relay in this version.
    """
    mdf_fraction = post_metrics.average_flex_mdf / baseline_metrics.average_flex_mdf
    continue_needed = mdf_fraction < MDF_RECOVERY_THRESHOLD
    return continue_needed, mdf_fraction


def main():
    byb_serial = None
    esp32_serial = None

    print("Backyard Brains to ESP32-S3 stimulation bridge")
    print("----------------------------------------------")
    print(f"BYB input:      {BYB_PORT} at {BYB_BAUD} baud")
    print(f"ESP32 output:   {ESP32_PORT} at {ESP32_BAUD} baud")
    print()
    print("Close Spike Recorder and Arduino Serial Monitor before running.")
    print()

    try:
        esp32_serial = open_serial_port(ESP32_PORT, ESP32_BAUD, "ESP32-S3 output")
        if esp32_serial is None:
            return 1

        print("\nWaiting 2 seconds for the ESP32-S3 to reset after opening serial...")
        time.sleep(2.0)

        # Fail-safe default: relay OFF before baseline assessment begins.
        send_relay_command(esp32_serial, "OFF")

        byb_serial = open_serial_port(BYB_PORT, BYB_BAUD, "Backyard Brains input")
        if byb_serial is None:
            return 1

        sample_reader = BYBSampleReader(byb_serial)

        baseline_metrics = run_assessment_block(
            sample_reader,
            "Initial baseline assessment",
        )

        if baseline_metrics.average_flex_mdf <= 0:
            raise RuntimeError("Initial baseline flex MDF was not usable.")

        cycle_number = 1

        while True:
            run_stimulation_cycle(esp32_serial, sample_reader, cycle_number)

            post_metrics = run_assessment_block(
                sample_reader,
                f"Post-stimulation assessment cycle {cycle_number}",
                baseline_metrics=baseline_metrics,
            )

            continue_needed, mdf_fraction = should_continue_stimulation(
                post_metrics,
                baseline_metrics,
            )

            print()
            print("=" * 72)
            print(f"Decision after cycle {cycle_number}")
            print("=" * 72)
            print(
                f"Average flex MDF is {mdf_fraction * 100.0:.1f}% "
                f"of baseline."
            )
            print(
                f"Continue threshold is below "
                f"{MDF_RECOVERY_THRESHOLD * 100.0:.1f}% of baseline MDF."
            )

            if continue_needed:
                print("Decision: CONTINUE stimulation.")
                print(
                    "Reason: post-cycle flex MDF is still below the recovery "
                    "threshold."
                )
                cycle_number += 1
            else:
                print("Decision: STOP stimulation.")
                print(
                    "Reason: post-cycle flex MDF has recovered to the target "
                    "threshold or higher."
                )
                send_relay_command(esp32_serial, "OFF")
                print("Session complete. Relay remains OFF.")
                break

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Stopping immediately...")
        safe_send_off(esp32_serial)

    except SerialException as error:
        print(f"\n\nSerial communication error: {error}")
        print("Switching relay OFF and exiting.")
        safe_send_off(esp32_serial)

    except Exception as error:
        print(f"\n\nERROR: {error}")
        print("Switching relay OFF and exiting.")
        safe_send_off(esp32_serial)
        print_byb_data_help()
        return 1

    finally:
        safe_send_off(esp32_serial)

        if byb_serial is not None and byb_serial.is_open:
            byb_serial.close()

        if esp32_serial is not None and esp32_serial.is_open:
            esp32_serial.close()

        print("Serial ports closed. Goodbye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
