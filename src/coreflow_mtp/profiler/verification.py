"""Component fit and leave-context-out checks; not engine-capacity validation."""
import csv
import json
from pathlib import Path

import numpy as np

from coreflow_mtp.predictor.ar import ARLatencyPredictor
from coreflow_mtp.predictor.linear import LinearRegression


def error_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    residual = np.abs(predicted - actual)
    return {"points": len(actual), "mae_seconds": float(residual.mean()),
            "wmape": float(residual.sum() / max(np.abs(actual).sum(), 1e-15)),
            "p95_absolute_seconds": float(np.quantile(residual, .95))}


def verify_profile(bundle, output):
    checked = ARLatencyPredictor(bundle)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    datasets = {}
    data = json.loads(checked.profile_paths["attn_profile.json"].read_text())
    for phase in ("decode", "prefill"):
        datasets[phase] = [{"x": [int(b), int(n)], "y": seconds,
                            "context": int(n) // int(b) if phase == "decode" else int(n)}
                          for b, rows in data[phase].items() for n, seconds in rows.items()]
    data = json.loads(checked.profile_paths["interference_profile.json"].read_text())
    for b, rows in data.items():
        datasets[f"interference_B{b}"] = [{"x": [int(l), int(s)], "y": seconds, "context": int(l)}
            for l, shorts in rows.items() for s, seconds in shorts.items()]
    metrics, predictions = {}, []
    for name, rows in datasets.items():
        x, y = np.asarray([r["x"] for r in rows]), np.asarray([r["y"] for r in rows])
        fit = LinearRegression().fit(x, y)
        training = fit.predict(x)
        if name.startswith("interference"):
            training = np.maximum(training, 0)
        metrics[name] = {"training": error_metrics(y, training), "rank": fit.rank_}
        held_y, held_p = [], []
        for context in sorted({r["context"] for r in rows}):
            train = [i for i, r in enumerate(rows) if r["context"] != context]
            test = [i for i, r in enumerate(rows) if r["context"] == context]
            if len(train) < 3:
                continue
            model = LinearRegression().fit(x[train], y[train])
            predicted = model.predict(x[test])
            if name.startswith("interference"):
                predicted = np.maximum(predicted, 0)
            for i, p in zip(test, predicted):
                predictions.append({"component": name, "held_context": context, "x0": x[i, 0],
                    "x1": x[i, 1], "actual_seconds": y[i], "predicted_seconds": p})
            held_y.extend(y[test])
            held_p.extend(predicted)
        metrics[name]["leave_context_out"] = error_metrics(held_y, held_p) if held_y else None
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "held_context_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "held_context", "x0", "x1",
                                                    "actual_seconds", "predicted_seconds"])
        writer.writeheader()
        writer.writerows(predictions)
    report = ["# K=0 组件模型拟合检查", "", "这里检验组件拟合，不是整引擎吞吐或分配收益。",
        "训练误差与留出完整 context 后的误差分开报告；干扰存在0值，使用 WMAPE 与绝对误差，不对0值强行计算逐点百分比。", "",
        "| 组件 | 训练 WMAPE | 留 context WMAPE | 留 context MAE（µs） |",
        "| --- | ---: | ---: | ---: |"]
    for name, values in metrics.items():
        held = values["leave_context_out"]
        report.append(f"| {name} | {values['training']['wmape']:.2%} | " +
            (f"{held['wmape']:.2%} | {held['mae_seconds'] * 1e6:.3f} |" if held else "不可用 | 不可用 |"))
    report.extend(["", "小网格 smoke 只用于验证 profiling→建模接口，不是论文正式标定。",
                   "原始带符号差值和截距位于 profiling 的 raw_samples.json；本表的干扰目标保留原实现 max(0, overhead) 定义。",
                   "完整性能验收仍需真实并发到达 workload，不能把单条 workflow 的倒数当成可持续吞吐。"])
    (out / "拟合检查_zh.md").write_text("\n".join(report) + "\n")
    return metrics
