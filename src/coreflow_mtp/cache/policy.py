"""Paper keep-alive decision, calibrated swap costs, and bounded tool history."""
import json
import math
from pathlib import Path


def keepalive_decision(tool_seconds, swap_out_seconds, swap_in_seconds):
    costs = [swap_out_seconds, swap_in_seconds]
    if any(not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("invalid measured swap cost")
    keep = (tool_seconds is not None and math.isfinite(tool_seconds) and tool_seconds > 0
            and swap_out_seconds > tool_seconds - swap_in_seconds)
    if tool_seconds is not None and (not math.isfinite(tool_seconds) or tool_seconds < 0):
        raise ValueError("invalid expected tool latency")
    return {"mode": "predictive", "keep_cache_in_gpu": keep,
            "retain_seconds": tool_seconds if keep else 0.,
            "expected_tool_seconds": tool_seconds, "swap_out_seconds": swap_out_seconds,
            "swap_in_seconds": swap_in_seconds,
            "reason": "swap_cost_exceeds_tool_gap" if keep else "cpu_retention_preferred"}


class CachePolicyProfile:
    def __init__(self, d2h, h2d, tool_samples=None, provenance=None):
        self.fits = {"d2h": d2h, "h2d": h2d}
        self.tool_samples = {}
        self.provenance = provenance
        for fit in self.fits.values():
            for key in ("intercept_seconds", "seconds_per_byte"):
                if not math.isfinite(fit[key]) or fit[key] < 0:
                    raise ValueError("invalid calibrated swap cost")
        for key, values in (tool_samples or {}).items():
            for value in values:
                self.observe(*key.split("/", 1), value)

    @classmethod
    def load(cls, path, model, speculative_k, tp):
        value = json.loads(Path(path).read_text())
        if value.get("kind") == "coreflow_ar_components":
            from coreflow_mtp.predictor.ar import ARLatencyPredictor
            checked = ARLatencyPredictor(path, speculative_k=speculative_k)
            if tp != 1 or Path(model).resolve() != Path(checked.geometry.model).resolve():
                raise ValueError("cache-policy profile model/TP mismatch")
            swap = checked.swap_profile
            return cls(swap["d2h"], swap["h2d"], provenance={"bundle": str(Path(path).resolve()),
                "scope": swap.get("scope", "raw_copy_estimate_not_measured_RPC_cost")})
        if (value.get("kind") != "coreflow_cache_policy" or not value.get("provenance")
                or Path(value["model"]).resolve() != Path(model).resolve()
                or value.get("speculative_k") != speculative_k or value.get("tensor_parallel_size") != tp):
            raise ValueError("cache policy requires calibrated matching model/K/TP provenance")
        return cls(value["d2h"], value["h2d"], value.get("tool_samples"), value["provenance"])

    def observe(self, agent, tool, seconds):
        if not agent or not tool or "/" in agent or "/" in tool:
            raise ValueError("tool observation requires simple nonempty agent/tool IDs")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid observed tool duration")
        key = f"{agent}/{tool}"
        values = self.tool_samples.setdefault(key, [])
        values.append(seconds)
        del values[:-128]
        return {"samples": len(values), "mean_seconds": sum(values) / len(values)}

    def request_policy(self, agent, tool, expected_seconds=None):
        values = self.tool_samples.get(f"{agent}/{tool}", [])
        predicted = expected_seconds if expected_seconds is not None else (
            sum(values) / len(values) if values else None)
        return {"tool_seconds": predicted, **self.fits,
                "provenance": self.provenance, "tool_samples": len(values)}


def decide_from_bytes(policy, copy_bytes):
    costs, sources = {}, {}
    for key in ("d2h", "h2d"):
        fit = policy[key]
        bounds = fit.get("valid_bytes")
        if bounds is not None and not bounds[0] <= copy_bytes <= bounds[1]:
            if key not in policy.get("fallbacks", {}):
                return {"mode": "predictive", "keep_cache_in_gpu": False, "retain_seconds": 0.,
                    "expected_tool_seconds": policy["tool_seconds"], "estimated_copy_bytes": copy_bytes,
                    "reason": "uncalibrated_copy_size_cpu_fallback",
                    "cost_provenance": policy.get("provenance")}
            fit = policy["fallbacks"][key]
            sources[key] = "raw_copy_prior_outside_native_calibration_domain"
        else:
            sources[key] = "native_measured" if bounds is not None else "supplied_calibrated_prior"
        costs[key] = fit["intercept_seconds"] + fit["seconds_per_byte"] * copy_bytes
    return {**keepalive_decision(policy["tool_seconds"], costs["d2h"], costs["h2d"]),
            "estimated_copy_bytes": copy_bytes, "cost_provenance": policy.get("provenance"),
            "cost_sources": sources}
