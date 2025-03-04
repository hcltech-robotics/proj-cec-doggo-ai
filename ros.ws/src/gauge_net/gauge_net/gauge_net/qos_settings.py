from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos_overriding_options import QoSOverridingOptions, QoSPolicyKind


GAUGE_QOS_OVERRIDE = QoSOverridingOptions(
    policy_kinds=[
        QoSPolicyKind.HISTORY,
        QoSPolicyKind.DEPTH,
        QoSPolicyKind.RELIABILITY,
        QoSPolicyKind.DURABILITY,
    ]
)

GAUGE_QOS_PROFILE = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,  # Only keep the latest message
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
