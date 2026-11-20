"""Đánh giá MỘT LẦN trên tập test độc lập sau khi đã chốt checkpoint bằng val (yêu cầu 2).

Chạy:
    python research/prognosis/evaluate.py --data $DATA --manifest research/prognosis/split_manifest.csv \
        --runs research/prognosis/runs

Ghi: backend/ai_core/weights/risk_reference.json, docs/cai_tien/ket_qua_test.md, docs/cai_tien/figures/*.png,
runs/test_report.json (đánh dấu đã đọc test; chạy lại cần --force và sẽ ghi rõ trong báo cáo).

--rehearse chạy toàn bộ quy trình trên tập VAL (không đụng test) và chỉ ghi vào runs/rehearsal/, để bắt lỗi trước lần thật.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from sksurv.metrics import integrated_brier_score  # noqa: E402
from sksurv.util import Surv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from survival_metrics import bootstrap_ci, breslow_baseline, calibration_table, predict_survival  # noqa: E402
from train import ROOT, WEIGHTS_DIR, load_patients, predict, select_split  # noqa: E402

sys.path.insert(0, str(ROOT / "backend"))
from ai_core.architectures.survival_net import MultimodalBrainTumorModel, c_index_score  # noqa: E402
from ai_core.feature_schema import SCHEMA_VERSION, load_rna_reference  # noqa: E402

CALIBRATION_TIMES = (12, 24)
CALIBRATION_TOLERANCE = 0.10
CURVE_TIMES = list(range(0, 37, 6))
GROUP_NAMES = ("Low", "Medium", "High")
BASELINE_FEATURES = ("age", "gender", "idh_mutation", "mgmt_methylation")
DOCS = ROOT / "docs" / "cai_tien"


def _load_model(path: Path, num_genes: int, device: str) -> MultimodalBrainTumorModel:
    model = MultimodalBrainTumorModel(num_genes=num_genes)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def load_members(runs: Path, selection: dict, num_genes: int, device: str) -> dict[str, MultimodalBrainTumorModel]:
    """Nạp các mô hình dùng để dự đoán: v2 (selection có "version" 2) là mọi seed của ensemble, v1 là seed đã chọn."""
    if selection.get("version", 1) >= 2:
        return {str(s): _load_model(runs / f"seed{s}" / selection["checkpoint"], num_genes, device) for s in selection["ensemble_seeds"]}
    chosen = selection["chosen_seed"]
    return {str(chosen): _load_model(runs / f"seed{chosen}" / "best.pth", num_genes, device)}


def ensemble_predict(models, patients, num_genes: int, device: str) -> np.ndarray:
    """Risk = trung bình log-hazard của các mô hình (một mô hình thì là chính nó)."""
    return np.mean([predict(m, patients, num_genes, device) for m in models], axis=0)


def previous_reports(prognosis_dir: Path) -> list[dict]:
    """Mọi lần đánh giá test đã ghi (research/prognosis/runs*/test_report.json), cũ nhất trước; khóa chung cho v1 và v2."""
    reports = [json.loads(path.read_text()) for path in prognosis_dir.glob("runs*/test_report.json")]
    return sorted(reports, key=lambda r: r["evaluated_at"])


def _c_index(risk, patients) -> float:
    return float(c_index_score(np.asarray(risk), np.array([p.time for p in patients]), np.array([p.event for p in patients])))


def clinical_baseline(data_dir: Path, train_ids, test_ids) -> float:
    """CoxPH tuyến tính chỉ dùng tuổi, gender, IDH, MGMT (thiếu → trung vị train + cờ thiếu), fit trên train."""
    df = pd.read_csv(data_dir / "dataset_multimodal_final.csv").set_index("patient_id")
    features = pd.DataFrame(index=df.index)
    medians = df.loc[train_ids, list(BASELINE_FEATURES)].median()
    for col in BASELINE_FEATURES:
        features[col] = df[col].fillna(medians[col])
        if df.loc[train_ids, col].isna().any():
            features[f"{col}_missing"] = df[col].isna().astype(float)
    # Cột hằng số trên train (vd. idh_mutation thật chỉ có giá trị 0) làm ma trận suy biến: bỏ, giữ cờ thiếu.
    features = features.loc[:, features.loc[train_ids].std() > 0]
    features["time"], features["event"] = df["time"], df["event"]
    fitter = CoxPHFitter(penalizer=0.1).fit(features.loc[train_ids], duration_col="time", event_col="event")
    test = features.loc[test_ids]
    return float(c_index_score(fitter.predict_log_partial_hazard(test).to_numpy(), test.time.to_numpy(), test.event.to_numpy()))


def integrated_brier(train_patients, test_patients, test_surv: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    """IBS (IPCW) của mô hình và của Kaplan–Meier không đặc trưng (fit trên train) trên cùng lưới thời gian."""
    as_surv = lambda ps: Surv.from_arrays([bool(p.event) for p in ps], [p.time for p in ps])  # noqa: E731
    km = KaplanMeierFitter().fit([p.time for p in train_patients], [p.event for p in train_patients])
    km_curve = np.tile(km.survival_function_at_times(times).to_numpy(), (len(test_patients), 1))
    survival_train, survival_test = as_surv(train_patients), as_surv(test_patients)
    return (float(integrated_brier_score(survival_train, survival_test, test_surv, times)),
            float(integrated_brier_score(survival_train, survival_test, km_curve, times)))


def _plots(test_patients, test_groups, pred_by_time, calib_tables, grid, s0, test_risk, out_docs: Path) -> None:
    figures = out_docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for t, table in calib_tables.items():
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Hiệu chuẩn hoàn hảo")
        ax.scatter(table.predicted_mean, table.km_observed, s=60, zorder=3)
        for row in table.itertuples():
            ax.annotate(f"{GROUP_NAMES[row.group]} (n={row.n})", (row.predicted_mean, row.km_observed), textcoords="offset points", xytext=(6, -10))
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"Dự đoán S({t} tháng)", ylabel=f"Kaplan–Meier quan sát S({t} tháng)",
               title=f"Calibration trên tập test – {t} tháng")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(figures / f"calibration_{t}m.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    times = np.array([p.time for p in test_patients])
    events = np.array([p.event for p in test_patients])
    for group, name in enumerate(GROUP_NAMES):
        members = test_groups == group
        if not members.any():
            continue
        km = KaplanMeierFitter().fit(times[members], events[members], label=f"{name} – KM quan sát (n={members.sum()})")
        km.plot_survival_function(ax=ax, ci_show=False)
        curve = [float(np.mean(predict_survival(test_risk[members], grid, s0, t))) for t in CURVE_TIMES]
        ax.plot(CURVE_TIMES, curve, "--", color=ax.get_lines()[-1].get_color(), label=f"{name} – dự đoán trung bình")
    ax.set(xlim=(0, 48), ylim=(0, 1), xlabel="Tháng", ylabel="Xác suất sống còn", title="Tập test: KM theo nhóm nguy cơ vs dự đoán")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "km_by_risk_group_test.png", dpi=150)
    plt.close(fig)


def _markdown(report: dict) -> str:
    m = report["test_metrics"]
    lines = [
        "# Kết quả tiên lượng trên tập test độc lập (Yêu cầu 2)",
        "",
        f"Đánh giá lúc {report['evaluated_at']}, sau khi đã chốt mô hình mà không đọc tập test. "
        f"Train {report['n_train']} ca, test {report['n_test']} ca (`research/prognosis/split_manifest.csv`).",
        f"Mô hình triển khai: {report['model_desc']}.",
        "",
        "## Chỉ số chính",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---|",
        f"| C-index test (mô hình triển khai) | **{m['c_index']:.4f}** (KTC95% bootstrap {m['c_index_ci'][0]:.4f}–{m['c_index_ci'][1]:.4f}, {report['bootstrap']} mẫu) |",
        f"| C-index test từng seed ({len(report['seed_test_c_index'])} seed) | {np.mean(list(report['seed_test_c_index'].values())):.4f} ± {np.std(list(report['seed_test_c_index'].values())):.4f} |",
        f"| Integrated Brier score (IPCW, {report['ibs_times'][0]:.0f}–{report['ibs_times'][1]:.0f} tháng) | {m['ibs']:.4f} (Kaplan–Meier không đặc trưng: {m['ibs_km']:.4f}) |",
        f"| Đối chứng CoxPH lâm sàng (tuổi, gender, IDH, MGMT) | C-index test {report['clinical_baseline_c_index']:.4f} |",
        f"| C-index val cũ (checkpoint 0,6936, chọn theo val, không có test) | không so trực tiếp được: khác split, khác dữ liệu WSI |",
        "",
        "## Ablation trên tập test (tắt mô thức lúc suy luận)",
        "",
        "| Cấu hình | C-index test |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.4f} |" for name, value in report["ablation"].items()]
    lines += ["", "## Calibration (nhóm theo tam phân vị risk của tập train)", ""]
    for t, rows in report["calibration"].items():
        lines += [f"### {t} tháng", "", "| Nhóm | n | Dự đoán TB | KM quan sát | Chênh lệch |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {GROUP_NAMES[r['group']]} | {r['n']} | {r['predicted_mean']:.3f} | {r['km_observed']:.3f} | {r['abs_diff']:.3f} |" for r in rows]
        lines.append("")
    verdict = "ĐẠT" if report["calibrated"] else "KHÔNG ĐẠT"
    lines += [
        f"## Tiêu chí hiển thị xác suất sống còn: **{verdict}**",
        "",
        f"Tiêu chí chốt trước khi train: tại 12 và 24 tháng, mọi nhóm có |dự đoán − KM| ≤ {CALIBRATION_TOLERANCE:.2f}, "
        "và IBS mô hình < IBS Kaplan–Meier.",
        "Không đạt thì giao diện chỉ hiển thị điểm nguy cơ tương đối (percentile so với tập train) kèm khung giới hạn.",
        "",
        "Hình: `figures/calibration_12m.png`, `figures/calibration_24m.png`, `figures/km_by_risk_group_test.png`.",
    ]
    if report.get("rerun"):
        lines += ["", f"## Các lần đánh giá test trước", "",
                  f"Đây là lần đánh giá test thứ {report['rerun'] + 1}. Quy trình chọn mô hình lần này được chốt trước khi chạy "
                  "(`docs/cai_tien/ke_hoach_v2.md`) và không dùng kết quả test lần trước để chọn.", "",
                  "| Lần | Thời điểm | Mô hình | C-index test (KTC95%) |", "|---:|---|---|---|"]
        for i, prev in enumerate(report["previous"], start=1):
            c = prev["test_metrics"]
            lines.append(f"| {i} | {prev['evaluated_at']} | {prev.get('model_desc') or 'v1: seed ' + str(prev['chosen_seed']) + ', epoch ' + str(prev['chosen_epoch'])} "
                         f"| {c['c_index']:.4f} ({c['c_index_ci'][0]:.4f}–{c['c_index_ci'][1]:.4f}) |")
        lines.append(f"| {report['rerun'] + 1} | {report['evaluated_at']} | {report['model_desc']} | {m['c_index']:.4f} ({m['c_index_ci'][0]:.4f}–{m['c_index_ci'][1]:.4f}) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="đánh giá lại test (sẽ ghi chú trong báo cáo)")
    parser.add_argument("--rehearse", action="store_true", help="chạy thử trên tập val, không đụng test")
    args = parser.parse_args()

    out_docs = args.runs / "rehearsal" if args.rehearse else DOCS
    out_docs.mkdir(parents=True, exist_ok=True)
    marker = (out_docs if args.rehearse else args.runs) / "test_report.json"
    previous = [] if args.rehearse else previous_reports(Path(__file__).resolve().parent)
    if previous and not args.force:
        raise SystemExit(f"Tập test đã được đánh giá ({previous[-1]['evaluated_at']}). Dùng --force nếu thật sự cần.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = json.loads((args.runs / "selection.json").read_text())
    manifest = pd.read_csv(args.manifest)
    rna_matrix = pd.read_csv(args.data / "PROCESSED_RNA" / "rna_matrix.csv", index_col=0)
    rna_ref = load_rna_reference(str(args.runs / "rna_reference.npz"))
    num_genes = len(rna_ref.columns)
    train_rows = select_split(manifest, tuple(selection.get("train_splits", ["train"])))
    eval_split = "val" if args.rehearse else "test"
    test_rows = manifest[manifest.split == eval_split].reset_index(drop=True)  # chỗ DUY NHẤT đọc tập test
    train_patients = load_patients(args.data, train_rows, rna_matrix, rna_ref)
    test_patients = load_patients(args.data, test_rows, rna_matrix, rna_ref)

    members = load_members(args.runs, selection, num_genes, device)
    risk_of = lambda ps: ensemble_predict(list(members.values()), ps, num_genes, device)  # noqa: E731
    if selection.get("version", 1) >= 2:
        model_desc = (f"v2: ensemble {len(members)} seed, {selection['epochs']} epoch, {selection['num_genes']} cột RNA "
                      f"(cấu hình {selection['gene_config']}, C-index CV {selection['cv_c_index']:.4f})")
    else:
        model_desc = (f"v1: seed {selection['chosen_seed']}, epoch {selection['chosen_epoch']} "
                      f"(chọn theo C-index val = {selection['val_c_index'][str(selection['chosen_seed'])]:.4f})")
    train_risk = risk_of(train_patients)
    test_risk = risk_of(test_patients)
    grid, s0 = breslow_baseline(train_risk, [p.time for p in train_patients], [p.event for p in train_patients])

    c_index = _c_index(test_risk, test_patients)
    c_ci = bootstrap_ci(lambda idx: _c_index(test_risk[idx], [test_patients[i] for i in idx]), size=len(test_patients), n=args.bootstrap)
    seed_c = {}
    if selection.get("version", 1) >= 2:
        for seed, member in members.items():
            seed_c[seed] = _c_index(predict(member, test_patients, num_genes, device), test_patients)
    else:
        for seed_dir in sorted(args.runs.glob("seed*")):
            seed_c[seed_dir.name.removeprefix("seed")] = _c_index(
                predict(_load_model(seed_dir / "best.pth", num_genes, device), test_patients, num_genes, device), test_patients)

    cuts = np.quantile(train_risk, [1 / 3, 2 / 3])
    test_groups = np.digitize(test_risk, cuts)
    pred_by_time = {t: predict_survival(test_risk, grid, s0, t) for t in CALIBRATION_TIMES}
    times_test = np.array([p.time for p in test_patients])
    events_test = np.array([p.event for p in test_patients])
    calib = {t: calibration_table(pred_by_time[t], times_test, events_test, test_groups, t) for t in CALIBRATION_TIMES}

    ibs_grid = np.arange(3, 37, dtype=float)
    ibs_grid = ibs_grid[(ibs_grid > times_test.min()) & (ibs_grid < times_test.max())]
    test_surv = np.stack([predict_survival(test_risk, grid, s0, t) for t in ibs_grid], axis=1)
    ibs, ibs_km = integrated_brier(train_patients, test_patients, test_surv, ibs_grid)

    ablation = {"Đầy đủ mô thức": c_index}
    for name, field in (("Tắt WSI", "wsi"), ("Tắt RNA", "rna")):
        ablation[name] = _c_index(risk_of([replace(p, **{field: None}) for p in test_patients]), test_patients)
    ablation["Tắt cả WSI và RNA"] = _c_index(risk_of([replace(p, wsi=None, rna=None) for p in test_patients]), test_patients)

    calibrated = all(table.abs_diff.max() <= CALIBRATION_TOLERANCE for table in calib.values()) and ibs < ibs_km
    _plots(test_patients, test_groups, pred_by_time, calib, grid, s0, test_risk, out_docs)

    test_metrics = {"c_index": c_index, "c_index_ci": list(c_ci), "ibs": ibs, "ibs_km": ibs_km}
    risk_reference = {
        "schema_version": SCHEMA_VERSION, "train_risk_sorted": sorted(float(r) for r in train_risk),
        "tertile_cuts": [float(c) for c in cuts], "s0": {"t": grid.tolist(), "s": s0.tolist()},
        "calibrated": bool(calibrated), "test_metrics": test_metrics,
        "n_train": len(train_patients), "n_test": len(test_patients),
    }
    if not args.rehearse:
        (WEIGHTS_DIR / "risk_reference.json").write_text(json.dumps(risk_reference), encoding="utf-8")

    report = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"), "rerun": len(previous), "model_desc": model_desc,
        "previous": [{k: prev.get(k) for k in ("evaluated_at", "model_desc", "chosen_seed", "chosen_epoch", "test_metrics")} for prev in previous],
        "n_train": len(train_patients), "n_test": len(test_patients), "bootstrap": args.bootstrap,
        "test_metrics": test_metrics, "seed_test_c_index": seed_c, "clinical_baseline_c_index": clinical_baseline(args.data, train_rows.patient_id, test_rows.patient_id),
        "ablation": ablation, "calibration": {t: table.to_dict("records") for t, table in calib.items()},
        "ibs_times": [float(ibs_grid[0]), float(ibs_grid[-1])], "calibrated": bool(calibrated),
    }
    marker.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_docs / "ket_qua_test.md").write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))


if __name__ == "__main__":
    main()
