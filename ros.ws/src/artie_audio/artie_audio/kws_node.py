#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData

from precise_lite_runner import PreciseLiteListener, ReadWriteStream

class KWS(Node):
    def __init__(self):
        super().__init__("precise_kws")

        # --- params ---
        self.declare_parameter("model_path", "")
        self.declare_parameter("sensitivity", 0.7)   # 0..1
        self.declare_parameter("trigger_level", 2)   # frames above threshold to fire
        self.declare_parameter("frame_len", 1600)    # samples per step (100 ms @ 16 kHz)

        model = self.get_parameter("model_path").get_parameter_value().string_value
        sensitivity = float(self.get_parameter("sensitivity").value)
        trigger_level = int(self.get_parameter("trigger_level").value)
        self.frame_len = int(self.get_parameter("frame_len").value)

        # --- ROS I/O ---
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=50)
        self.create_subscription(AudioData, "audio/pcm", self.on_audio, qos)
        self.pub_kws = self.create_publisher(String, "voice/kws", 10)

        # --- Precise-Lite: stream + listener ---
        self.stream = ReadWriteStream()  
        self.listener = PreciseLiteListener(
            model=model,
            stream=self.stream,
            chunk_size=self.frame_len,      # in S16 samples
            trigger_level=trigger_level,
            sensitivity=sensitivity,
            on_activation=self._on_hotword, # callback when hotword confirmed
        )
        self.listener.start()

        self.get_logger().info(
            f"Precise-lite ready: model={model}, sens={sensitivity}, trig={trigger_level}, frame={self.frame_len}"
        )

    def on_audio(self, msg: AudioData):
        # msg.data is uint8[] 
        b = bytes(msg.data)
        if b:
            self.stream.write(b)  # feed bytes directly; listener handles framing

    def _on_hotword(self):
        self.get_logger().info("Wake word detected")
        self.pub_kws.publish(String(data="wake"))

    def destroy_node(self):
        try:
            self.listener.stop()
        finally:
            super().destroy_node()

def main():
    rclpy.init()
    n = KWS()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
