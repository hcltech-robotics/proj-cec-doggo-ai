#!/usr/bin/env python3

import queue
import sys
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from builtin_interfaces.msg import Time as TimeMsg

from audio_common_msgs.msg import AudioData, AudioInfo

import sounddevice as sd


class MicStreamer(Node):
    def __init__(self) -> None:
        super().__init__("mic_streamer")

        # ---------- Parameters ----------
        self.declare_parameter("audio.input_device", "default")
        self.declare_parameter("audio.sample_rate", 16000)
        self.declare_parameter("audio.channels", 1)
        self.declare_parameter("audio.frame_ms", 20)

        self.input_device: str = self.get_parameter("audio.input_device").get_parameter_value().string_value
        self.sample_rate: int = self.get_parameter("audio.sample_rate").get_parameter_value().integer_value
        self.channels: int = self.get_parameter("audio.channels").get_parameter_value().integer_value
        self.frame_ms: int = self.get_parameter("audio.frame_ms").get_parameter_value().integer_value

        if self.channels not in (1, 2):
            self.get_logger().warn(f"audio.channels={self.channels} is unusual; forcing mono for ASR/KWS.")
            self.channels = 1

        if self.sample_rate not in (8000, 16000, 32000, 44100, 48000):
            self.get_logger().warn(f"audio.sample_rate={self.sample_rate} looks odd; 16000 is recommended.")
        if self.frame_ms not in (10, 20, 30):
            self.get_logger().warn(f"audio.frame_ms={self.frame_ms} not typical; using {self.frame_ms} ms anyway.")

        frames_per_block = int(self.sample_rate * (self.frame_ms / 1000.0))
        bytes_per_frame = self.channels * 2  # S16LE = 2 bytes per sample
        self.expected_bytes = frames_per_block * bytes_per_frame

        # ---------- Publishers ----------
        audio_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        info_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # latched
        )

        self.pub_audio = self.create_publisher(AudioData, "audio/pcm", audio_qos)
        self.pub_info = self.create_publisher(AudioInfo, "audio/info", info_qos)

        # ---------- Audio stream ----------
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)
        self.stream: Optional[sd.RawInputStream] = None

        self._open_stream(frames_per_block)

        # publish info once (latched)
        self._publish_audio_info()

        # Timer: drain queue and publish chunks
        self.timer = self.create_timer(0.005, self._drain)

        self.get_logger().info(
            f"MicStreamer up — device='{self.input_device}', {self.sample_rate} Hz, "
            f"{self.channels} ch, {self.frame_ms} ms frames (~{self.expected_bytes} bytes/msg)."
        )

    def _open_stream(self, frames_per_block: int) -> None:
        """Open the sounddevice RawInputStream with the requested params."""
        dev = None if self.input_device == "default" else self.input_device

        def _callback(indata, frames, time_info, status):
            # status includes xruns/overflows; log occasionally
            if status:
                # Avoid log spam: only warn every so often
                self.get_logger().debug(f"audio status: {status}")
            try:
                # indata is a bytes-like object (int16 little-endian)
                chunk: bytes = bytes(indata)
                if len(chunk) != self.expected_bytes:
                    # truncate or pad
                    if len(chunk) > self.expected_bytes:
                        chunk = chunk[: self.expected_bytes]
                    else:
                        chunk = chunk + b"\x00" * (self.expected_bytes - len(chunk))
                self._q.put_nowait(chunk)
            except queue.Full:
                # drop on overload; BEST_EFFORT semantics upstream
                pass

        try:
            self.stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=frames_per_block,
                dtype="int16",
                channels=self.channels,
                device=dev,
                callback=_callback,
            )
            self.stream.start()
        except Exception as e:
            self.get_logger().error(
                f"Failed to open audio device '{self.input_device}': {e}. "
                "If using Pulse/PipeWire, ensure you're running as your user and that a default source exists."
            )
            raise

    def _publish_audio_info(self) -> None:
        info = AudioInfo()

        info.sample_rate = int(self.sample_rate)
        info.channels = int(self.channels)
        info.sample_format = "S16LE"  # PCM signed 16-bit little-endian
        info.bitrate = int(16 * self.sample_rate * self.channels)  # bits/sec
        info.coding_format = "PCM"

        self.pub_info.publish(info)

    def _drain(self) -> None:
        # Publish as many queued chunks as we have (keeps latency low)
        while not self._q.empty():
            chunk = self._q.get_nowait()
            msg = AudioData()
            try:
                msg.data = chunk                 # fast path
            except TypeError:
                msg.data = list(chunk)           # fallback for old distros
            try:
                self.pub_audio.publish(msg)
            except Exception as e:
                self.get_logger().warn(f"publish error: {e}")
                break

    def destroy_node(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        super().destroy_node()


def main(argv=None):
    rclpy.init(args=argv)
    node = MicStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
