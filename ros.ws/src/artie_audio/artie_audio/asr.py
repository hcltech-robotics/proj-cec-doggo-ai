#!/usr/bin/env python3
from dataclasses import dataclass
import json
import threading
import time
from typing import Optional, List

import numpy as np
from scipy.signal import resample_poly

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
)

from audio_common_msgs.msg import AudioData, AudioInfo
from go2_interfaces.msg import WebRtcReq
from std_msgs.msg import String
from std_srvs.srv import Trigger

from faster_whisper import WhisperModel


@dataclass(frozen=True)
class VUI_COLOR:

    WHITE: str = 'white'
    RED: str = 'red'
    YELLOW: str = 'yellow'
    BLUE: str = 'blue'
    GREEN: str = 'green'
    CYAN: str = 'cyan'
    PURPLE: str = 'purple'


def s16le_bytes_to_mono_f32(data: bytes, channels: int) -> np.ndarray:
    """Convert interleaved S16LE PCM bytes to mono float32 [-1, 1]."""
    pcm = np.frombuffer(data, dtype=np.int16)
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return (pcm.astype(np.float32)) / 32768.0


class FastWhisperListener(Node):
    def __init__(self):
        super().__init__("fast_whisper_listener")

        # ---------- Parameters ----------
        self.declare_parameter("listen_seconds", 7)                  # how many seconds to RECORD after trigger
        self.declare_parameter("whisper.model", "small")             # tiny, base, small, medium, large-v3, distil-*, etc.
        self.declare_parameter("whisper.device", "auto")             # "cuda", "cpu", or "auto"
        self.declare_parameter("whisper.compute_type", "float16")    # "float16" (GPU) or "int8" CPU
        self.declare_parameter("language", "en")                     # "" = auto; e.g., "hu", "en"
        self.declare_parameter("kws.topic", "/voice/kws")                   # where wake words arrive as String("wake")

        self.listen_seconds: int = self.get_parameter("listen_seconds").get_parameter_value().integer_value
        self.model_name: str = self.get_parameter("whisper.model").get_parameter_value().string_value
        self.device_param: str = self.get_parameter("whisper.device").get_parameter_value().string_value
        self.compute_type: str = self.get_parameter("whisper.compute_type").get_parameter_value().string_value
        self.language: str = self.get_parameter("language").get_parameter_value().string_value
        self.kws_topic: str = self.get_parameter("kws.topic").get_parameter_value().string_value

        # Filled after AudioInfo
        self.sample_rate: Optional[int] = None
        self.channels: Optional[int] = None

        # ---- Post-trigger capture state ----
        self._recording: bool = False
        self._capture_buf: List[bytes] = []
        self._record_lock = threading.Lock()
        self._record_timer: Optional[threading.Timer] = None

        # For the Trigger service: wait until transcription finishes
        self._srv_wait_event: Optional[threading.Event] = None
        self._last_transcript: str = ""

        # Reentrant callback group so service + subs can run concurrently
        self._cb_group = ReentrantCallbackGroup()

        # Publishers
        out_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_text = self.create_publisher(String, "asr/text", out_qos)

        self.pub_led_color = self.create_publisher(WebRtcReq, "/webrtc_req", 10)
        self._led_timer = None
        self._led_is_blue = True

        # Subscribers
        info_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        audio_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # match your MicStreamer or change both to RELIABLE
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.sub_info = self.create_subscription(
            AudioInfo, "audio/info", self._on_info, info_qos, callback_group=self._cb_group
        )
        self.sub_audio = self.create_subscription(
            AudioData, "audio/pcm", self._on_audio, audio_qos, callback_group=self._cb_group
        )
        self.sub_kws = self.create_subscription(
            String, self.kws_topic, self._on_kws, out_qos, callback_group=self._cb_group
        )

        # Service for manual trigger (records then returns transcript)
        self.srv_trigger = self.create_service(
            Trigger, "asr/transcribe_now", self._on_trigger_srv, callback_group=self._cb_group
        )

        # Load Whisper model
        device = self._choose_device(self.device_param)
        self.get_logger().info(f"Device: {device}, Compute type: {self.compute_type}")
        try:
            self.model = WhisperModel(self.model_name, device=device, compute_type=self.compute_type)
            self.get_logger().info(f"Loaded faster-whisper model='{self.model_name}' on {device} ({self.compute_type}).")
        except Exception as e:
            self.get_logger().error(f"Failed to load faster-whisper model '{self.model_name}': {e}")
            raise

        self.get_logger().info("FastWhisperListener ready. Waiting for audio/info...")

    # ---------- Subscriptions ----------
    def _on_info(self, msg: AudioInfo):
        self.sample_rate = msg.sample_rate
        self.channels = max(1, msg.channels) if msg.channels in (1, 2) else 1
        self.get_logger().info(f"audio/info: {self.sample_rate} Hz, {self.channels} ch")

    def _on_audio(self, msg: AudioData):
        # only store when _recording is True
        if not self._recording:
            return
        chunk = bytes(msg.data)
        with self._record_lock:
            self._capture_buf.append(chunk)

    def _on_kws(self, msg: String):
        if msg.data.strip().lower() == "wake":
            if not self._recording:
                self.get_logger().info(f"Wake detected → recording next {self.listen_seconds}s...")
                self._start_recording(async_publish=True)  # non-blocking path

    # ---------- Service ----------
    def _on_trigger_srv(self, request, response):
        # synchronous record → transcribe → reply
        if self._recording:
            self.get_logger().warn("Already recording; rejecting new trigger.")
            response.success = False
            response.message = "Already recording."
            return response

        self.get_logger().info(f"Service trigger → recording next {self.listen_seconds}s...")
        self._srv_wait_event = threading.Event()
        self._start_recording(async_publish=False)

        # Wait while other callbacks (audio) can run on other threads
        finished = self._srv_wait_event.wait(self.listen_seconds + 30)
        if not finished:
            response.success = False
            response.message = "Timed out waiting for transcription."
            return response

        response.success = True
        response.message = self._last_transcript if self._last_transcript else "(no speech detected)"
        return response

    # ---------- Recording control ----------
    def _start_recording(self, async_publish: bool):
        with self._record_lock:
            self._recording = True
            self._capture_buf = []
        # one-shot timer to finalize capture
        if self._record_timer is not None:
            try:
                self._record_timer.cancel()
            except Exception:
                pass
        self._record_timer = threading.Timer(
            self.listen_seconds, self._finalize_recording, kwargs={"async_publish": async_publish}
        )
        self._record_timer.start()
        self._start_led_pulse()

    def _finalize_recording(self, async_publish: bool):
        # Stop capturing and grab the buffer
        with self._record_lock:
            self._recording = False
            raw = b"".join(self._capture_buf)
            self._capture_buf = []
        self._stop_led_pulse()

        if self._record_timer is not None:
            try:
                self._record_timer.cancel()
            except Exception:
                pass
            self._record_timer = None

        if not raw:
            self.get_logger().warn("No audio captured during recording window.")
            self._publish_and_signal("")
            return

        # Transcribe in background if async_publish (KWS path), else do it here (service path)
        if async_publish:
            t = threading.Thread(target=self._transcribe_bytes_and_publish, args=(raw,), daemon=True)
            t.start()
        else:
            text = self._transcribe_bytes(raw)
            self._publish_and_signal(text)

    # ---------- Transcription ----------
    def _transcribe_bytes_and_publish(self, raw: bytes):
        text = self._transcribe_bytes(raw)
        self._publish_and_signal(text)

    def _transcribe_bytes(self, raw: bytes) -> str:
        sr = self.sample_rate or 16000
        ch = self.channels or 1

        # bytes → mono float32
        wav = s16le_bytes_to_mono_f32(raw, ch)

        # resample to 16 kHz if needed
        if sr != 16000:
            wav = resample_poly(wav, 16000, sr)

        # Run faster-whisper
        text = ""
        try:
            segments, info = self.model.transcribe(
                wav,
                beam_size=1,
                language=self.language if self.language else None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=200),
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            self.get_logger().info(
                f"ASR ({self.model_name}): \"{text}\"  (lang={getattr(info, 'language', None) or 'auto'})"
            )
        except Exception as e:
            self.get_logger().error(f"ASR failed: {e}")
        return text

    def _publish_and_signal(self, text: str):
        # Publish text on topic
        msg = String()
        msg.data = text
        self.pub_text.publish(msg)

        # Signal waiting service (if any)
        self._last_transcript = text
        if self._srv_wait_event is not None:
            self._srv_wait_event.set()
            self._srv_wait_event = None

    # ---------- LED pulsing helpers ----------
    def _start_led_pulse(self):
        try:
            self._led_tick()
        except Exception:
            pass
        if self._led_timer is not None:
            try:
                self._led_timer.cancel()
            except Exception:
                pass
        self._led_timer = self.create_timer(1.0, self._led_tick)

    def _stop_led_pulse(self):
        if self._led_timer is not None:
            try:
                self._led_timer.cancel()
            except Exception:
                pass
            self._led_timer = None
        # switch back to green at the end
        self._set_led_color(VUI_COLOR.GREEN, time=1)

    def _led_tick(self):
        # alternate BLUE <-> PURPLE
        color = VUI_COLOR.BLUE if self._led_is_blue else VUI_COLOR.PURPLE
        self._led_is_blue = not self._led_is_blue
        # set time=1 so it fades before next tick
        self._set_led_color(color=color, time=2)


    # ---------- Utils ----------
    def _choose_device(self, device_param: str) -> str:
        if device_param in ("cuda", "cpu"):
            return device_param
        try:
            import torch  # optional
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cuda"

    def _set_led_color(self, color: VUI_COLOR, time=5, flash_cycle=None):
        req = WebRtcReq()
        req.id = 0  
        req.topic = 'rt/api/vui/request'   
        req.api_id = 1007
        req.priority = 0  
        p = {}
        p["color"] = color
        p["time"] = time
        if flash_cycle:
            p["flash_cycle"] = flash_cycle
        
        req.parameter = json.dumps(p)  

        self.get_logger().info(f'Setting LED to {color} | payload={req.parameter}')
        self.pub_led_color.publish(req)


def main():
    rclpy.init()
    node = FastWhisperListener()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
