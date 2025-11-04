#!/usr/bin/env python3
import os, json, base64, io, time, re
from typing import Dict, Any, Callable, Optional
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from openai import OpenAI
from go2_interfaces.msg import WebRtcReq
import numpy as np
from artie_chatgpt.tools.tool_selector import ToolSelector
from sentence_transformers import SentenceTransformer
from std_srvs.srv import Empty


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Use this tool to move the dog forward ~1 meter (short pulse).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_backward",
            "description": "Use this tool to move the dog backward ~1 meter (short pulse).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_left",
            "description": "Use this tool to strafe left ~1 meter (short pulse).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_right",
            "description": "Use this tool to strafe right ~1 meter (short pulse).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": "Rotate in place a small amount (short pulse).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Immediately stop all motion.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dance",
            "description": "Use this tool to perform a dance.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "standup",
            "description": "Reset to neutral standing pose (stand on 4 legs).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lie_down",
            "description": "Lie down / prone posture.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sit",
            "description": "Sit posture.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hello",
            "description": "Say hello (wave / greet / high five).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_analyze",
            "description": "Take a photo ahead and analyze it; return a textual summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_tts",
            "description": "Stop talking",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "company_search",
            "description": "Search internal company FAQs/policies to answer factual questions about the org.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

# ---------------------------- System prompt ----------------------------
SYSTEM_PROMPT = """You are Artie, a ROS2-connected assistant for live demos.
- Be concise, concrete, and safe. If a user asks you to do something that looks unsafe, refuse and explain briefly.
- If the user asks for an action (navigate, wag_tail, sit_down, wave_leg, dance, etc.), decide parameters sensibly and call the corresponding tool.
- If the user asks about the company or policies, call company_search(query) first and use those facts. If unknown, say so.
- If the user asks about the scene (e.g., ‘what do you see’, ‘describe what’s ahead’), call image_analyze and use its result in your answer. If no image is available, say so briefly.
- Otherwise, answer normally without tools.
- Prefer metric units. Yaw is in radians. Always confirm action intents in your wording.
"""

FIELD_SYNONYMS = {
    "location": "locations",
    "site": "locations",
    "macro geography": "macro geography",
    "macro_geography": "macro geography",
    "people": "no. of people",
    "headcount": "no. of people",
    "languages": "services provided in languages:",
    "clients": "key clients",
}

# ---------------------------- The ROS 2 Node ----------------------------
class OpenAIOrchestrator(Node):
    def __init__(self):
        super().__init__("openai_orchestrator")
        self.bridge = CvBridge()
        self.tools = ToolSelector(self)
        self.tool_registry = self.tools.registry()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Config
        self.pkg_share = get_package_share_directory("artie_chatgpt")
        self.company_kb_path   = os.path.join(self.pkg_share, "resource", "info.json")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") 

        # In-memory vectordb
        self.embed_model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._kb_texts: list[str] = []
        self._kb_metas: list[dict] = []
        self._kb_vecs: np.ndarray | None = None
        self._encoder = SentenceTransformer(self.embed_model_name, device="cuda")      
        self.company_kb = self._load_company_kb(self.company_kb_path)

        
        # WebRTC publisher to Go2
        self.webrtc_pub = self.create_publisher(WebRtcReq, "/webrtc_req", 10)
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.llm_pub = self.create_publisher(String, "/llm/response", 10)
        self.tts_stop_cli = self.create_client(Empty, "/tts/stop")
        self.MAX_LIN = 1.2             # m/s clamp
        self.MAX_YAW = 1.5             # rad/s clamp


        # Keep latest image
        self.latest_image_b64 = None
        self.subscription_asr = self.create_subscription(String, "/asr/text", self.on_asr_text, 10)
        self.subscription_img = self.create_subscription(Image, "/camera/color/image_raw", self.on_image, 10)

        self.get_logger().info(f"OpenAI model: {self.model}")
        if not os.getenv("OPENAI_API_KEY"):
            self.get_logger().warn("OPENAI_API_KEY is not set!")

        self.get_logger().info(f"vec mean norm: {np.linalg.norm(self._kb_vecs, axis=1).mean():.3f}")

    # ---------------- service call ----------------
    def stop_tts(self, timeout_sec: float = 0.5) -> None:
        try:
            if not self.tts_stop_cli.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().warn(
                    f"/tts/stop not available after {timeout_sec:.1f}s — sending empty /llm/response as fallback."
                )
                # Fallback: empty text to stop
                self.llm_pub.publish(String(data=""))
                return

            req = Empty.Request()
            fut = self.tts_stop_cli.call_async(req)

            def _done(_):
                if fut.result() is not None:
                    self.get_logger().info("TTS stop acknowledged by /tts/stop.")
                else:
                    self.get_logger().error(f"TTS stop failed: {fut.exception()}")

            fut.add_done_callback(_done)

        except Exception as e:
            self.get_logger().error(f"stop_tts() error: {e}")
            # Last-resort fallback
            self.llm_pub.publish(String(data=""))


    # ---------------- helpers ----------------

    def _norm_meta(self, meta: dict) -> dict:
        if not meta:
            return {}
        out = {}
        for k, v in meta.items():
            k_norm = " ".join(k.strip().lower().split())
            v_norm = " ".join(str(v).strip().split())
            out[k_norm] = v_norm
        return out

    def _apply_synonyms(self, meta_norm: dict) -> dict:
        out = {}
        for k, v in meta_norm.items():
            out[FIELD_SYNONYMS.get(k, k)] = v
        return out

    def _meta_to_text(self, meta: dict) -> str:
        return "\n".join(f"{k}: {meta[k]}" for k in sorted(meta.keys()))

    def _parse_filters(self, query: str):
        pattern = r'([A-Za-z][A-Za-z _.:/-]{0,50}):\s*([^:]+?)(?=\s+[A-Za-z][A-Za-z _.:/-]{0,50}:\s*|$)'
        pairs = [(m.group(1).strip().lower(), m.group(2).strip().lower()) for m in re.finditer(pattern, query)]
        free = query
        for k, v in pairs:
            free = re.sub(re.escape(k) + r'\s*:\s*' + re.escape(v), '', free, flags=re.IGNORECASE)
        return pairs, " ".join(free.split())

    # --------------- loader ------------------
    def _load_company_kb(self, path: str) -> list:
        try:
            raw = json.load(open(path, "r", encoding="utf-8"))

            kb, texts, metas = [], [], []
            for row in raw:
                meta_raw = row.get("metadata", {}) or {}
                meta_norm = self._apply_synonyms(self._norm_meta(meta_raw))
                q = (meta_raw.get("question") or "").strip() if "question" in meta_raw else ""
                a = (meta_raw.get("answer") or "").strip() if "answer" in meta_raw else ""

                meta_text = self._meta_to_text(meta_norm)
                page = (row.get("pageContent") or "").strip()
                text = "\n".join(p for p in [meta_text, page, f"Q: {q}\nA: {a}" if q or a else ""] if p).strip()

                entry = {
                    "id": row.get("id"),
                    "q": q,
                    "a": a,
                    "meta": meta_norm,
                    "text": text,
                    "text_lc": text.lower(),
                }
                kb.append(entry)
                texts.append(text)
                metas.append({"id": entry["id"], "q": q, "a": a, "meta": meta_norm})

            self.company_kb = kb                     
            self._kb_texts, self._kb_metas = texts, metas
            self._kb_vecs = self._embed_texts(texts)   # build vectors once
            self.get_logger().info(f"Loaded company KB: {len(kb)} entries | vecs: {None if self._kb_vecs is None else self._kb_vecs.shape}")
            return kb
        except Exception as e:
            self.get_logger().warn(f"KB not loaded: {e}")
            self.company_kb, self._kb_texts, self._kb_metas, self._kb_vecs = [], [], [], None
            return []

    # --------------- embeddings ----------------
    def _embed_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray | None:
        if not texts:
            return None

        emb = self._encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return emb

    # --------------- search ----------------
    def company_search(self, query: str, k: int = 5) -> dict:
        q = (query or "").strip()
        if not q:
            return {"matches": []}

        filters, free_text = self._parse_filters(q)

        candidates = list(range(len(self.company_kb)))
        if filters:
            cand = []
            for idx in candidates:
                meta = self.company_kb[idx]["meta"]
                ok = True
                for key, val in filters:
                    key_syn = FIELD_SYNONYMS.get(key, key)
                    if key_syn in meta:
                        ok = ok and (val in meta[key_syn].lower())
                    else:
                        ok = ok and any(val in (mv or "").lower() for mv in meta.values())
                    if not ok: break
                if ok: cand.append(idx)
            candidates = cand

        if not candidates:
            candidates = list(range(len(self.company_kb)))

        # vector rank
        if self._kb_vecs is not None and len(self._kb_texts) == self._kb_vecs.shape[0]:
            q_text = free_text or q
            qv = self._encoder.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

            sims = (self._kb_vecs[candidates] @ qv)     # cosine (normalized)
            top_rel = np.argpartition(sims, -k)[-k:]
            top_rel = top_rel[np.argsort(sims[top_rel])[::-1]]
            top_idx = [candidates[i] for i in top_rel]
        else:
            # keyword fallback
            q_lc = (free_text or q).lower()
            toks = [t for t in re.split(r"[^a-z0-9]+", q_lc) if t]
            scored = []
            for idx in candidates:
                txt = self.company_kb[idx]["text_lc"]
                tok_hits = sum(txt.count(t) for t in set(toks))
                phrase = 2 if q_lc and q_lc in txt else 0
                scored.append((tok_hits + phrase, idx))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_idx = [i for _, i in scored[:k]]

        # format results
        out = []
        for i in top_idx:
            e = self.company_kb[i]
            meta = e["meta"]
            if e["q"] and e["a"]:
                out.append(f"{e['q']} → {e['a']}")
            else:
                fields = []
                for key in ["macro geography", "locations", "description", "no. of people", "key clients"]:
                    if key in meta:
                        fields.append(f"{key}: {meta[key]}")
                out.append("; ".join(fields) if fields else e["text"])

        return {"matches": out}


    # ---------- Image handling ----------
    def on_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            import cv2
            _, buf = cv2.imencode(".jpg", cv_img)  # compress a bit
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            self.latest_image_b64 = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")

    def image_analyze(self):
        if not self.latest_image_b64:
            return {"ok": False, "error": "no_image"}

        msgs = [
            {"role":"system","content":"Describe the scene concisely; mention hazards first."},
            {"role":"user","content":[
                {"type":"text","text":"What do you see?"},
                {"type":"image_url","image_url":{"url": self.latest_image_b64}},
            ]}
        ]
        r = self.openai_client.chat.completions.create(model=self.model, messages=msgs, temperature=0.2)
        return {"ok": True, "analysis": r.choices[0].message.content.strip()}


    # ---------- ASR handler ----------
    def on_asr_text(self, msg: String):
        user_text = msg.data.strip()
        if not user_text:
            return
        self.get_logger().info(f"[ASR] {user_text}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "You have access to function tools. Call them when actions or facts are needed."},
            {"role": "user", "content": user_text},
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.4,
        )

        result_text, _ = self._handle_tool_calls_and_finalize(messages, response)
        self.get_logger().info(f"[LLM] {result_text}")

        msg = String()
        msg.data = result_text
        self.llm_pub.publish(msg)

    # ---------- Tool loop + finalize ----------
    def _handle_tool_calls_and_finalize(self, messages, response):
        """Executes tools if requested, then gets the final answer from the model."""
        # Accumulate assistant messages and tool outputs
        latest = response.choices[0].message
        messages.append({"role": "assistant", "content": latest.content or "", "tool_calls": latest.tool_calls})

        # If there are tool calls, execute them and append results
        if latest.tool_calls:
            for call in latest.tool_calls:
                fn = call.function.name
                args = json.loads(call.function.arguments or "{}")
                self.get_logger().info(f"[toolcall] {fn}({args})")

                impl = self.tool_registry.get(fn)
                if not impl:
                    tool_output = {"error": f"Unknown tool: {fn}"}
                else:
                    try:
                        tool_output = impl(**args)
                    except Exception as e:
                        tool_output = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": fn,
                    "content": json.dumps(tool_output),
                })

            # Second pass: ask the model to produce a user-facing answer
            final = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
            )
            text = final.choices[0].message.content.strip()
            return text, final

        # No tools needed: return the original assistant text
        text = (latest.content or "").strip()
        return text, response


def main():
    rclpy.init()
    node = OpenAIOrchestrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
