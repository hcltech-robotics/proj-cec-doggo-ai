#!/usr/bin/env python3
import os
import threading
import queue
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from std_srvs.srv import Empty
from elevenlabs.play import stream as play_stream


class ElevenLabsTTSNode(Node):

    def __init__(self):
        super().__init__("tts_node")

        self.declare_parameter("voice_id", "21m00Tcm4TlvDq8ikWAM")   # e.g. Rachel UUID
        self.declare_parameter("model_id", "eleven_monolingual_v2")  # or eleven_multilingual_v2
        self.declare_parameter("api_key_env", "ELEVENLABS_API_KEY")
        self.declare_parameter("output_format", "mp3_44100_128")     # doc-friendly default
        self.declare_parameter("interrupt_on_new_message", True)

        self.voice_id = self.get_parameter("voice_id").get_parameter_value().string_value
        self.model_id = self.get_parameter("model_id").get_parameter_value().string_value
        self.api_key_env = self.get_parameter("api_key_env").get_parameter_value().string_value
        self.output_format = self.get_parameter("output_format").get_parameter_value().string_value
        self.interrupt = self.get_parameter("interrupt_on_new_message").get_parameter_value().bool_value

        # ---- ElevenLabs client ----
        api_key = os.getenv(self.api_key_env, "")
        self.client = ElevenLabs(api_key=api_key or None)
        if not api_key:
            self.get_logger().warn(
                f"No ElevenLabs API key found in env '{self.api_key_env}'. "
                "Public voices may still work; private voices require an API key."
            )

        # ---- ROS I/O ----
        self.sub = self.create_subscription(String, "/llm/response", self._on_text, 10)
        self.srv_stop = self.create_service(Empty, "/tts/stop", self._on_stop)
        self.msg_q: "queue.Queue[str]" = queue.Queue(maxsize=8)
        self.stop_flag = threading.Event()       # to stop current playback between utterances
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info("ElevenLabs TTS ready. Subscribed to /llm/response")

    def _is_bytes_like(self, x):
        return isinstance(x, (bytes, bytearray, memoryview))


    # ---------- ROS callback ----------
    def _on_stop(self, request, response):
        self.get_logger().info("Stop service called")
        self.stop_flag.set()
        with self.msg_q.mutex:
            self.msg_q.queue.clear()
        return response

    def _on_text(self, msg: String):
        text = (msg.data or "").strip().lower()

        if text == "":
            self.get_logger().info("Stop command — halting playback and clearing queue.")
            self.stop_flag.set()
            with self.msg_q.mutex:
                self.msg_q.queue.clear()
            return

        if self.interrupt:
            self.stop_flag.set()  # ensures any current playback will halt between chunks
        try:
            self.msg_q.put_nowait(text)
        except queue.Full:
            try:
                _ = self.msg_q.get_nowait()
            except queue.Empty:
                pass
            self.msg_q.put_nowait(text)


    # ---------- worker ----------
    def _worker_loop(self):
        while rclpy.ok():
            try:
                text = self.msg_q.get(timeout=0.25)
            except queue.Empty:
                continue

            if self.interrupt and self.stop_flag.is_set():
                latest = text
                while True:
                    try:
                        latest = self.msg_q.get_nowait()
                    except queue.Empty:
                        break
                self.stop_flag.clear()
                text = latest  # speak only the newest once

            self._speak(text)


    def _drain_queue_keep_latest(self, current_text: str) -> None:
        """When interrupting, drop older entries and keep only the most recent."""
        latest = current_text
        while True:
            try:
                latest = self.msg_q.get_nowait()
            except queue.Empty:
                break
        # requeue only the latest for immediate playback
        try:
            self.msg_q.put_nowait(latest)
        except queue.Full:
            pass

    # ---------- ElevenLabs call ----------
    def _speak(self, text: str):
        """
        Interruptible playback using elevenlabs.play.stream().
        Works whether convert() returns bytes or an iterator of chunks.
        """
        self.get_logger().info(f'TTS: "{text[:80]}{"…" if len(text) > 80 else ""}"')
        self.stop_flag.clear()

        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=self.output_format,  # e.g. "mp3_44100_128" or "wav_44100"
            )

            # Case A: iterator/generator — pass through with a stop guard
            if hasattr(audio, "__iter__") and not self._is_bytes_like(audio):
                def guarded_iter():
                    for chunk in audio:
                        if self.stop_flag.is_set():
                            break
                        yield chunk
                play_stream(guarded_iter())
                return

            # Case B: bytes-like — slice into chunks so stop takes effect mid-utterance
            view = memoryview(audio)
            CHUNK = 4096
            def chunks():
                i, n = 0, len(view)
                while i < n and not self.stop_flag.is_set():
                    yield view[i:i+CHUNK]
                    i += CHUNK
            play_stream(chunks())

        except Exception as e:
            self.get_logger().error(f"ElevenLabs TTS failed: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = ElevenLabsTTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
