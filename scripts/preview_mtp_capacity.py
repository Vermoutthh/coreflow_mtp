#!/usr/bin/env python3
"""CPU-only code-review entry; never start a GPU or modify an allocation.

--inspect-existing prints supported component predictions, not measured capacity.
--demo checks arithmetic with explicitly ILLUSTRATIVE inputs, not fitted results.
--scenario builds fixed modes from the frozen models and calls the production
ThroughputEstimator MTP entry; unknown mixed yield/resources stay unsupported.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "coreflow_mtp/src"))
from coreflow_mtp.allocator.throughput_estimator import (
    ThroughputEstimator, MTPExecutionMode, MTPRequestStats, MTPResourceLimits)
from coreflow_mtp.predictor.mtp import MTPProfilePredictor, UnsupportedMTPProfile


def illustrative_demo():
    """Same numbers as section 8 of the design note, NOT an experiment."""
    stats = MTPRequestStats(mean_prefill_tokens=512, mean_output_tokens=96)
    limits = MTPResourceLimits(4096, 32, {"illustrative_pool": 100}, 1000)
    ar = MTPExecutionMode(name="illustrative_AR", k=0, decode_batch_size=4,
        decode_seconds=.040, decode_tokens_per_round=4, mixed_decode_batch_size=4,
        mixed_seconds=.080, mixed_decode_tokens_per_round=4,
        scheduled_prefill_tokens=512, effective_prefill_tokens_per_round=512,
        decode_required_pages={"illustrative_pool": 20}, mixed_required_pages={"illustrative_pool": 25},
        decode_workspace_bytes=100, mixed_workspace_bytes=200,
        profiling_supported=True, native_shape_supported=True, preemption_free=True,
        provenance="ARITHMETIC_DEMO_NOT_real_profiling_or_capacity")
    sd = replace(ar, name="illustrative_SD", k=3, decode_seconds=.060,
                 mixed_seconds=.100, decode_tokens_per_round=12, mixed_decode_tokens_per_round=12)
    results = {}
    for mode in (ar, sd):
        results[mode.name] = dict(at_1_5_requests_s=ThroughputEstimator.mtp_one_second_budget(1.5, stats, mode),
            search=ThroughputEstimator.analyze_mtp_throughput(stats, [mode], limits))
    return dict(scope="仅验证示意算术；所有成本、产出、资源均为示意值，不是实验结果", examples=results)


def inspect_existing(predictor):
    rows = []
    for agent in ("Plan", "Task"):
        for k in range(7):
            contexts = [1024] * 4
            row = dict(agent=agent, B=4, C=1024, K=k)
            try:
                row["decode"] = predictor.predict_decode(contexts, k)
                row["pure_yield"] = predictor.predict_pure_yield(agent, contexts, k)
                row["mixed_at_P256"] = predictor.predict_mixed(contexts, k, 256)
                row["component_status"] = "available"
            except UnsupportedMTPProfile as exc:
                row.update(component_status="partially_supported", reason=str(exc))
            row["capacity_status"] = "not_computed"
            rows.append(row)
    return dict(scope="已有模型组件预览；不是容量预测或准确率验证", components=rows,
        missing_for_real_capacity=["与执行配置匹配的native可达形状、物理页与workspace证据",
            "同一混合形状的有效prefill进度及decode产出；不能直接复用pure yield",
            "完整prefill阶段与异长context的适用范围"], provenance=predictor.provenance)


def build_mode(spec, predictor, agent, needs_prefill, excluded_workflows=()):
    k = spec["k"]
    contexts = spec["decode_contexts"]
    cost = predictor.predict_decode(contexts, k)
    pure_yield = predictor.predict_pure_yield(agent, contexts, k, excluded_workflows=excluded_workflows)
    fields = dict(name=spec["name"], k=k, decode_batch_size=len(contexts),
        decode_seconds=cost["total_seconds"], decode_tokens_per_round=pure_yield["total_decode_tokens_per_round"],
        decode_required_pages=spec.get("decode_required_pages", {}),
        decode_workspace_bytes=spec.get("decode_workspace_bytes"),
        profiling_supported=spec.get("configuration_transfer_reviewed", False),
        native_shape_supported=spec.get("native_shape_supported", False),
        preemption_free=spec.get("preemption_free", False), provenance=spec.get("provenance", ""))
    if needs_prefill:
        if spec.get("prefill_request_count", 1) != 1:
            raise UnsupportedMTPProfile("mixed cost was profiled with ONE newly prefilling request")
        mixed_contexts = spec["mixed_decode_contexts"]
        scheduled = spec["scheduled_prefill_tokens"]
        mixed_cost = predictor.predict_mixed(mixed_contexts, k, scheduled,
            prefill_history_before=spec.get("prefill_history_before", 0))
        if "mixed_decode_tokens_per_round" not in spec:
            raise UnsupportedMTPProfile("need an explicit mixed DECODE yield; pure yield is not a substitute")
        if spec.get("mixed_profile_covers_effective_cycle", False) is not True:
            raise UnsupportedMTPProfile("first cold-prefill chunk alone does not cover the effective mixed cycle")
        fields.update(mixed_decode_batch_size=len(mixed_contexts), mixed_seconds=mixed_cost["total_seconds"],
            mixed_decode_tokens_per_round=spec["mixed_decode_tokens_per_round"],
            scheduled_prefill_tokens=scheduled,
            effective_prefill_tokens_per_round=spec["effective_prefill_tokens_per_round"],
            prefill_request_count=spec.get("prefill_request_count", 1),
            mixed_required_pages=spec.get("mixed_required_pages", {}),
            mixed_workspace_bytes=spec.get("mixed_workspace_bytes"))
    return MTPExecutionMode(**fields)


def preview_scenario(data, predictor):
    stats = MTPRequestStats(**data["request_stats"])
    stats.validate()
    limits = MTPResourceLimits(**data["limits"])
    modes, rejected = [], []
    for spec in data["modes"]:
        try:
            modes.append(build_mode(spec, predictor, data["agent"], stats.mean_prefill_tokens > 0,
                                    data.get("excluded_workflows", ())))
        except UnsupportedMTPProfile as exc:
            rejected.append(dict(name=spec["name"], k=spec["k"], status="unsupported",
                                 capacity_requests_per_second=None, reasons=[str(exc)]))
    result = ThroughputEstimator.analyze_mtp_throughput(stats, modes, limits)
    result["candidates"].extend(rejected)
    result.update(agent=data["agent"], provenance=predictor.provenance,
        note="代码预览；没有读取实测容量作输入，没有执行GPU或修改allocator配置")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--demo", action="store_true")
    command.add_argument("--inspect-existing", action="store_true")
    command.add_argument("--scenario", type=Path)
    parser.add_argument("--cost-model", type=Path, default=ROOT / "coreflow_mtp/results/01_joint_validation/frozen_cost_model.json")
    parser.add_argument("--yield-profile", type=Path, default=ROOT / "coreflow_mtp/results/01_joint_validation/yield_profiles_full.csv")
    parser.add_argument("--mixed-model", type=Path, default=ROOT / "coreflow_mtp/results/01_mixed_rounds/mixed_20261102_2030_gpu0/analysis/model_fit/candidate_model.json")
    args = parser.parse_args()
    if args.demo:
        result = illustrative_demo()
    else:
        predictor = MTPProfilePredictor(args.cost_model, yield_path=args.yield_profile, mixed_path=args.mixed_model)
        result = inspect_existing(predictor) if args.inspect_existing else preview_scenario(json.loads(args.scenario.read_text()), predictor)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
