"""Engine-independent allocation interfaces; no vLLM/sklearn import at startup."""

from coreflow_mtp.allocator.instance_orchestrator import InstanceAssignment, InstanceOrchestrator

__all__ = ["InstanceAssignment", "InstanceOrchestrator"]
