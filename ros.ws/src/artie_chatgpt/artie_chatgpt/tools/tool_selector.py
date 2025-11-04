# artie_chatgpt/tools/tool_selector.py
from __future__ import annotations
import json, threading
from typing import Dict, Any, Callable, Protocol, Optional
from geometry_msgs.msg import Twist
from rclpy.node import Node
from go2_interfaces.msg import WebRtcReq


class ToolSelector:
    """Encapsulates robot actions exposed as LLM tools."""
    def __init__(self, node):
        self.node = node
        self._busy = False
        self._sport_api = {
            "dance":        1022,
            "standup": 1006,   
            "lie_down":     1005,
            "sit":          1009,
            "hello":        1016,
        }

    # ---------- helpers ----------
    def _clamp(self, v, lo, hi): return max(lo, min(hi, v))

    def _send_twist_once(self, lin_x=0.0, lin_y=0.0, yaw=0.0):
        t = Twist()
        t.linear.x  = self._clamp(lin_x, -self.node.MAX_LIN, self.node.MAX_LIN)
        t.linear.y  = self._clamp(lin_y, -self.node.MAX_LIN, self.node.MAX_LIN)
        t.angular.z = self._clamp(yaw,   -self.node.MAX_YAW, self.node.MAX_YAW)
        self.node.twist_pub.publish(t)
        return {"ok": True}

    def _send_sport_cmd(self, name: str):
        api_id = self._sport_api[name]
        req = WebRtcReq()
        req.id = 0
        req.topic = 'rt/api/sport/request'
        req.api_id = api_id
        req.priority = 0
        req.parameter = json.dumps({})
        self.node.webrtc_pub.publish(req)
        self.node.get_logger().info(f"[tool] {name} -> api_id={api_id}")
        return {"ok": True}

    # ---------- movement tools ----------
    def move_forward(self):  return self._send_twist_once(lin_x=+1.0)
    def move_backward(self): return self._send_twist_once(lin_x=-1.0)
    def move_left(self):     return self._send_twist_once(lin_y=+1.0)
    def move_right(self):    return self._send_twist_once(lin_y=-1.0)
    def rotate(self):        return self._send_twist_once(yaw=+0.8)
    def stop(self):          return self._send_twist_once(0,0,0)
         

    # ---------- sport/pose tools ----------
    def dance(self):         return self._send_sport_cmd("dance")
    def standup(self):  return self._send_sport_cmd("standup")
    def lie_down(self):      return self._send_sport_cmd("lie_down")
    def sit(self):           return self._send_sport_cmd("sit")
    def hello(self):         return self._send_sport_cmd("hello")

    # ---------- vision + KB + TTS ----------
    def image_analyze(self): return self.node.describe_latest_image()
    def company_search(self, query:str=""): return self.node.company_search(query)
    def stop_tts(self): return self.node.stop_tts()
    # ---------- registry exported to LLM ----------
    def registry(self) -> Dict[str, Callable]:
        return {
            "move_forward":  lambda **_: self.move_forward(),
            "move_backward": lambda **_: self.move_backward(),
            "move_left":     lambda **_: self.move_left(),
            "move_right":    lambda **_: self.move_right(),
            "rotate":        lambda **_: self.rotate(),
            "stop":          lambda **_: self.stop(),

            "dance":         lambda **_: self.dance(),
            "standup":  lambda **_: self.standup(),
            "lie_down":      lambda **_: self.lie_down(),
            "sit":           lambda **_: self.sit(),
            "hello":         lambda **_: self.hello(),

            "image_analyze": lambda **_: self.image_analyze(),
            "company_search": lambda query="": self.company_search(query),
            "stop_tts": lambda **_: self.stop_tts(),
        }
