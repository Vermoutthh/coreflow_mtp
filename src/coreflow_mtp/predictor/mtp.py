"""Read existing frozen native-MTP costs/yields. CPU only; never refit.

Costs are worker-cycle spans, not per-layer costs or HTTP queueing latency.
Mixed corrections are deliberately restricted to the observed shape points.
This adapter does not infer a mixed yield from a pure-decode yield table.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


class UnsupportedMTPProfile(ValueError):
    pass


class MTPProfilePredictor:
    def __init__(self, cost_path, *, yield_path=None, mixed_path=None):
        self.cost = json.loads(Path(cost_path).read_text())
        self.yields = {}
        self.provenance = {str(Path(cost_path).resolve()): self._sha(cost_path)}
        if yield_path:
            with Path(yield_path).open(encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    key = row["agent"], int(row["context_lo"]), int(row["K"]), int(row["B"])
                    if key in self.yields:
                        raise ValueError("duplicate yield profile")
                    self.yields[key] = dict(u=float(row["u"]), request_rounds=int(row["request_rounds"]),
                        workflows=int(row["workflows"]), train_workflow_ids=json.loads(row["train_workflow_ids"]))
            self.provenance[str(Path(yield_path).resolve())] = self._sha(yield_path)
        self.mixed = json.loads(Path(mixed_path).read_text()) if mixed_path else None
        self.mixed_shapes = set()
        if mixed_path:
            if self.mixed.get("units") != "milliseconds" or self.mixed["chosen"].get("aux_increment") != 0:
                raise UnsupportedMTPProfile("unreviewed mixed timing units or auxiliary increment")
            self.provenance[str(Path(mixed_path).resolve())] = self._sha(mixed_path)
            cases = Path(mixed_path).with_name("predictions.csv")
            if cases.exists():
                with cases.open(encoding="utf-8-sig") as stream:
                    for row in csv.DictReader(stream):
                        self.mixed_shapes.add((int(row["B"]), int(row["K"]), float(row["C"]), float(row["P_actual"])))
                self.provenance[str(cases.resolve())] = self._sha(cases)

    @staticmethod
    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _shape(contexts, k):
        if not isinstance(k, int) or isinstance(k, bool) or k < 0 or not contexts:
            raise ValueError("nonempty contexts and integer K >= 0 required")
        if any(not math.isfinite(c) or c <= 0 for c in contexts):
            raise ValueError("positive finite CURRENT full contexts required")
        return len(contexts), f"B{len(contexts)}_K{k}"

    def predict_decode(self, contexts, k):
        b, key = self._shape(contexts, k)
        if key not in self.cost["calibrated_C_bounds"]:
            raise UnsupportedMTPProfile(f"no pure-decode B/K calibration: {key}")
        low, high = self.cost["calibrated_C_bounds"][key]
        if any(c < low or c > high for c in contexts):
            raise UnsupportedMTPProfile(f"context outside [{low}, {high}] for {key}")
        # The cost calibration was equal-context. Do not silently claim a
        # same-mean heterogeneous batch has already been validated.
        if any(c != contexts[0] for c in contexts):
            raise UnsupportedMTPProfile("heterogeneous decode correction not validated")
        d, v = self.cost["draft"], self.cost["verification"]
        total = sum(contexts)
        s = (k + 1) * total + b * k * (k + 1) / 2
        draft = 0. if k == 0 else d["delta"] + k * (d[f"g_B{b}"] + d["beta_BC"] * total / 1000)
        verify = v[f"f_B{b}_K{k}"] + v["beta"] * s / 1000
        auxiliary = self.cost["auxiliary_BK"][key]
        if any(not math.isfinite(x) or x < 0 for x in (draft, verify, auxiliary)):
            raise ValueError("invalid predicted phase time; do not clip negative coefficients")
        return dict(draft_ms=draft, verification_ms=verify, auxiliary_ms=auxiliary,
                    total_seconds=(draft + verify + auxiliary) / 1000,
                    S_query_history=s, scope="equal_context_worker_cycle_NOT_HTTP_latency")

    def predict_pure_yield(self, agent, contexts, k, *, excluded_workflows=()):
        b, _ = self._shape(contexts, k)
        total, records = 0., []
        edges = (0, 2048, 4096, 8192, 16512)
        for c in contexts:
            lo = next((lo for lo, hi in zip(edges, edges[1:]) if lo <= c < hi), None)
            key = agent, lo, k, b
            if key not in self.yields:
                raise UnsupportedMTPProfile(f"no pure-decode semantic yield profile: {key}")
            record = self.yields[key]
            if set(excluded_workflows) & set(record["train_workflow_ids"]):
                raise UnsupportedMTPProfile("yield training overlaps requested holdout workflows")
            if not math.isfinite(record["u"]) or not 0 <= record["u"] <= k + 1:
                raise ValueError("invalid semantic yield profile")
            total += record["u"]
            records.append(record)
        return dict(total_decode_tokens_per_round=total, mean_per_request=total / b,
                    min_training_workflows=min(r["workflows"] for r in records),
                    scope="pure_decode_committed_tokens_NOT_accept_rate_or_mixed_yield")

    def predict_mixed(self, contexts, k, scheduled_prefill_tokens, *, prefill_history_before=0):
        base = self.predict_decode(contexts, k)
        if scheduled_prefill_tokens == 0:
            return base
        if self.mixed is None:
            raise UnsupportedMTPProfile("no mixed-cost model")
        support = self.mixed["support"]
        if (len(contexts) not in support["B"] or k not in support["K"]
                or any(c not in support["C"] for c in contexts) or prefill_history_before != 0):
            raise UnsupportedMTPProfile("mixed cost only profiled for B=1/4, C=1024, K=0/3/6, new prefill history=0")
        # The model JSON pools P across K; use its saved prediction cases to
        # retain the actual B/K/C/P association and hybrid-alignment details.
        if (len(contexts), k, float(contexts[0]), float(scheduled_prefill_tokens)) not in self.mixed_shapes:
            raise UnsupportedMTPProfile("actual prefill query shape not observed for this K")
        chosen = self.mixed["chosen"]
        if chosen["model"] != "plateau512":
            raise UnsupportedMTPProfile("unreviewed mixed-cost functional form")
        x, sd = scheduled_prefill_tokens / 1000, float(k > 0)
        beta = chosen["target_coefficients"]
        extra_v = beta[0] + beta[1] * sd + beta[2] * max(x - .512, 0.)
        extra_d = chosen["draft_coefficients"][0] * sd * x
        result = dict(base)
        result.update(draft_ms=base["draft_ms"] + extra_d,
                      verification_ms=base["verification_ms"] + extra_v,
                      total_seconds=base["total_seconds"] + (extra_d + extra_v) / 1000,
                      scope="exploratory_frozen_decode_plus_mixed_increment_NOT_capacity_validation")
        return result
