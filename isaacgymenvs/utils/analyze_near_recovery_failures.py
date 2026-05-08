import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import h5py
default_mpl_config = Path.home() / ".config" / "matplotlib"
if not os.access(default_mpl_config.parent, os.W_OK):
    fallback_cache = Path(__file__).resolve().parents[2] / ".matplotlib_cache"
    fallback_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(fallback_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(fallback_cache))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CATEGORY_NAME_TO_ID = {
    "afar": 0,
    "near_recovery": 1,
    "far_recovery": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze verified teacher bank success/failure patterns and emit plots."
    )
    parser.add_argument("--generation-assignment", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--category", default="near_recovery", choices=sorted(CATEGORY_NAME_TO_ID))
    parser.add_argument("--compare-run-dir", type=Path, default=None)
    parser.add_argument(
        "--compare-correct-side-only",
        action="store_true",
        help="Filter compare-run stored states to correct_region=True.",
    )
    return parser.parse_args()


def load_assignment(path):
    payload = json.loads(path.read_text())
    variants = {}
    for raw_variant in payload["variants"]:
        variant_id = int(raw_variant["variant_id"])
        variants[variant_id] = {
            "variant_id": variant_id,
            "object_id": int(raw_variant["object_id"]),
            "z_shift": float(raw_variant["z_shift"]),
            "table_x": float(raw_variant["table_size"][0]),
            "table_y": float(raw_variant["table_size"][1]),
            "object_mass": float(raw_variant.get("object_mass", np.nan)),
        }

    rank_env_to_variant = {}
    for entry in payload["entries"]:
        rank_env_to_variant[(int(entry["rank"]), int(entry["local_env_id"]))] = int(entry["variant_id"])
    return variants, rank_env_to_variant


def shard_paths(run_dir):
    return sorted(run_dir.glob("side_verified_teacher_states_rank*.hdf5"))


def load_attempt_stats(run_dir, category_id, variants, rank_env_to_variant):
    per_object = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0})
    per_variant = {}
    total_attempts = 0
    total_successes = 0

    for path in shard_paths(run_dir):
        rank = int(path.stem.split("rank")[-1])
        with h5py.File(path, "r") as handle:
            attempts = handle["attempt_counts"][category_id]
            successes = handle["success_counts"][category_id]
            for env_id in range(attempts.shape[0]):
                variant_id = rank_env_to_variant[(rank, env_id)]
                variant = variants[variant_id]
                attempt_count = int(attempts[env_id])
                success_count = int(successes[env_id])
                failure_count = attempt_count - success_count

                total_attempts += attempt_count
                total_successes += success_count

                per_object_row = per_object[variant["object_id"]]
                per_object_row["attempts"] += attempt_count
                per_object_row["successes"] += success_count
                per_object_row["failures"] += failure_count

                row = per_variant.setdefault(
                    variant_id,
                    {
                        "variant_id": variant_id,
                        "object_id": variant["object_id"],
                        "z_shift": variant["z_shift"],
                        "table_x": variant["table_x"],
                        "table_y": variant["table_y"],
                        "object_mass": variant["object_mass"],
                        "attempts": 0,
                        "successes": 0,
                        "failures": 0,
                    },
                )
                row["attempts"] += attempt_count
                row["successes"] += success_count
                row["failures"] += failure_count

    per_object_rows = []
    total_failures = max(total_attempts - total_successes, 1)
    for object_id, row in per_object.items():
        attempts = row["attempts"]
        successes = row["successes"]
        failures = row["failures"]
        per_object_rows.append(
            {
                "object_id": object_id,
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
                "success_rate": (successes / attempts) if attempts else 0.0,
                "failure_share": failures / total_failures,
            }
        )

    per_variant_rows = []
    for row in per_variant.values():
        attempts = row["attempts"]
        row = dict(row)
        row["success_rate"] = (row["successes"] / attempts) if attempts else 0.0
        per_variant_rows.append(row)

    per_object_rows.sort(key=lambda row: row["success_rate"])
    per_variant_rows.sort(key=lambda row: row["success_rate"])
    return {
        "total_attempts": total_attempts,
        "total_successes": total_successes,
        "overall_success_rate": (total_successes / total_attempts) if total_attempts else 0.0,
        "per_object_rows": per_object_rows,
        "per_variant_rows": per_variant_rows,
    }


def _compute_rel_base_xy(base_xyz, object_xy):
    dx = object_xy[:, 0] - base_xyz[:, 0]
    dy = object_xy[:, 1] - base_xyz[:, 1]
    yaw = base_xyz[:, 2]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rel_x = cos_yaw * dx + sin_yaw * dy
    rel_y = -sin_yaw * dx + cos_yaw * dy
    return rel_x, rel_y


def load_stored_states(
    run_dir,
    category_id,
    variants,
    rank_env_to_variant,
    success=True,
    correct_only=False,
):
    if success:
        count_key = "counts"
        joint_key = "joint_config"
        object_key = "object_center_world"
        region_key = "correct_region"
    else:
        count_key = "failure_counts"
        joint_key = "failure_joint_config"
        object_key = "failure_object_center_world"
        region_key = "failure_correct_region"

    rows = []
    for path in shard_paths(run_dir):
        rank = int(path.stem.split("rank")[-1])
        with h5py.File(path, "r") as handle:
            counts = handle[count_key][category_id]
            joint = handle[joint_key][category_id]
            object_center = handle[object_key][category_id]
            correct_region = handle[region_key][category_id].astype(bool)

            for env_id in range(counts.shape[0]):
                variant_id = rank_env_to_variant[(rank, env_id)]
                variant = variants[variant_id]
                count = int(counts[env_id])
                if count <= 0:
                    continue

                base_xyz = np.asarray(joint[env_id, :count, :3], dtype=np.float64)
                object_xyz = np.asarray(object_center[env_id, :count, :3], dtype=np.float64)
                correct = np.asarray(correct_region[env_id, :count], dtype=bool)
                if correct_only:
                    keep = correct
                    if not np.any(keep):
                        continue
                    base_xyz = base_xyz[keep]
                    object_xyz = object_xyz[keep]
                    correct = correct[keep]

                rel_x, rel_y = _compute_rel_base_xy(base_xyz, object_xyz[:, :2])
                for idx in range(base_xyz.shape[0]):
                    rows.append(
                        {
                            "variant_id": variant_id,
                            "object_id": variant["object_id"],
                            "base_x": float(base_xyz[idx, 0]),
                            "base_y": float(base_xyz[idx, 1]),
                            "base_yaw": float(base_xyz[idx, 2]),
                            "object_x": float(object_xyz[idx, 0]),
                            "object_y": float(object_xyz[idx, 1]),
                            "object_z": float(object_xyz[idx, 2]),
                            "obj_rel_base_x": float(rel_x[idx]),
                            "obj_rel_base_y": float(rel_y[idx]),
                            "correct_region": bool(correct[idx]),
                            "z_shift": variant["z_shift"],
                            "table_x": variant["table_x"],
                            "table_y": variant["table_y"],
                            "object_mass": variant["object_mass"],
                        }
                    )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_object_success_rates(path, per_object_rows):
    object_ids = [row["object_id"] for row in per_object_rows]
    success_rates = [row["success_rate"] for row in per_object_rows]
    attempts = [row["attempts"] for row in per_object_rows]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(rate) for rate in success_rates]
    ax1.bar(range(len(object_ids)), success_rates, color=colors)
    ax1.set_xticks(range(len(object_ids)))
    ax1.set_xticklabels(object_ids, rotation=90)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Success Rate")
    ax1.set_xlabel("Object ID")
    ax1.set_title("Near-Recovery Success Rate by Object")

    ax2 = ax1.twinx()
    ax2.plot(range(len(object_ids)), attempts, color="black", linewidth=1.5, marker="o", markersize=3)
    ax2.set_ylabel("Attempts")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_variant_histogram(path, per_variant_rows):
    success_rates = [row["success_rate"] for row in per_variant_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(success_rates, bins=40, color="#4477aa", edgecolor="white")
    ax.axvline(np.mean(success_rates), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(success_rates):.3f}")
    ax.axvline(np.median(success_rates), color="black", linestyle=":", linewidth=1.5, label=f"median={np.median(success_rates):.3f}")
    ax.set_xlabel("Variant Success Rate")
    ax.set_ylabel("Variant Count")
    ax.set_title("Near-Recovery Variant Success-Rate Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_variant_feature_scatter(path, per_variant_rows):
    success_rate = np.array([row["success_rate"] for row in per_variant_rows])
    z_shift = np.array([row["z_shift"] for row in per_variant_rows])
    table_y = np.array([row["table_y"] for row in per_variant_rows])
    table_x = np.array([row["table_x"] for row in per_variant_rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    scatter_kwargs = {"s": 16, "alpha": 0.45, "c": success_rate, "cmap": "viridis"}

    im = axes[0].scatter(z_shift, success_rate, **scatter_kwargs)
    axes[0].set_xlabel("z_shift")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Success vs z_shift")

    axes[1].scatter(table_y, success_rate, **scatter_kwargs)
    axes[1].set_xlabel("table_y")
    axes[1].set_title("Success vs table_y")

    axes[2].scatter(table_x, success_rate, **scatter_kwargs)
    axes[2].set_xlabel("table_x")
    axes[2].set_title("Success vs table_x")

    fig.colorbar(im, ax=axes, shrink=0.9, label="Success Rate")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_state_hexbin(path, success_rows, failure_rows):
    success_x = np.array([row["obj_rel_base_x"] for row in success_rows])
    success_y = np.array([row["obj_rel_base_y"] for row in success_rows])
    failure_x = np.array([row["obj_rel_base_x"] for row in failure_rows])
    failure_y = np.array([row["obj_rel_base_y"] for row in failure_rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, x, y, title in [
        (axes[0], success_x, success_y, "Successful Stored Starts"),
        (axes[1], failure_x, failure_y, "Failed Stored Starts"),
    ]:
        hb = ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("obj_rel_base_x")
        ax.set_ylabel("obj_rel_base_y")
        fig.colorbar(hb, ax=ax, label="Count")
    fig.suptitle("Correct-Only Near-Recovery Start Geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_failure_share_heatmap(path, success_rows, failure_rows):
    success_x = np.array([row["obj_rel_base_x"] for row in success_rows])
    success_y = np.array([row["obj_rel_base_y"] for row in success_rows])
    failure_x = np.array([row["obj_rel_base_x"] for row in failure_rows])
    failure_y = np.array([row["obj_rel_base_y"] for row in failure_rows])

    all_x = np.concatenate([success_x, failure_x])
    all_y = np.concatenate([success_y, failure_y])
    x_edges = np.linspace(all_x.min(), all_x.max(), 50)
    y_edges = np.linspace(all_y.min(), all_y.max(), 50)
    success_hist, _, _ = np.histogram2d(success_x, success_y, bins=(x_edges, y_edges))
    failure_hist, _, _ = np.histogram2d(failure_x, failure_y, bins=(x_edges, y_edges))
    total = success_hist + failure_hist
    failure_share = np.divide(failure_hist, total, out=np.full_like(failure_hist, np.nan), where=total > 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mesh = ax.pcolormesh(x_edges, y_edges, failure_share.T, vmin=0.0, vmax=1.0, cmap="coolwarm")
    ax.set_xlabel("obj_rel_base_x")
    ax.set_ylabel("obj_rel_base_y")
    ax.set_title("Failure Share by Start-State Geometry")
    fig.colorbar(mesh, ax=ax, label="failure / (success + failure)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_compare_success_geometry(path, current_success_rows, compare_success_rows):
    current_x = np.array([row["obj_rel_base_x"] for row in current_success_rows])
    current_y = np.array([row["obj_rel_base_y"] for row in current_success_rows])
    compare_x = np.array([row["obj_rel_base_x"] for row in compare_success_rows])
    compare_y = np.array([row["obj_rel_base_y"] for row in compare_success_rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, x, y, title in [
        (axes[0], compare_x, compare_y, "Mixed Bank Correct-Side Success"),
        (axes[1], current_x, current_y, "Correct-Only Success"),
    ]:
        hb = ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("obj_rel_base_x")
        ax.set_ylabel("obj_rel_base_y")
        fig.colorbar(hb, ax=ax, label="Count")
    fig.suptitle("Stored Successful Near-Recovery Starts")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(path, run_dir, stats, per_object_rows, per_variant_rows, success_rows, failure_rows):
    worst_objects = per_object_rows[:10]
    worst_variants = per_variant_rows[:15]
    success_rel_y = np.array([abs(row["obj_rel_base_y"]) for row in success_rows])
    failure_rel_y = np.array([abs(row["obj_rel_base_y"]) for row in failure_rows])

    lines = [
        f"run_dir: {run_dir}",
        f"overall_success_rate: {stats['overall_success_rate']:.4f}",
        f"total_attempts: {stats['total_attempts']}",
        f"total_successes: {stats['total_successes']}",
        f"total_failures: {stats['total_attempts'] - stats['total_successes']}",
        "",
        "worst_objects:",
    ]
    for row in worst_objects:
        lines.append(
            f"  object {row['object_id']}: success_rate={row['success_rate']:.3f} "
            f"attempts={row['attempts']} failures={row['failures']}"
        )
    lines.extend(
        [
            "",
            "worst_variants:",
        ]
    )
    for row in worst_variants:
        lines.append(
            f"  variant {row['variant_id']} object={row['object_id']} success_rate={row['success_rate']:.3f} "
            f"attempts={row['attempts']} z_shift={row['z_shift']:.3f} "
            f"table=({row['table_x']:.3f},{row['table_y']:.3f}) mass={row['object_mass']:.3f}"
        )
    lines.extend(
        [
            "",
            "stored_state_geometry:",
            f"  success |obj_rel_base_y| mean={success_rel_y.mean():.4f} p90={np.percentile(success_rel_y, 90):.4f}",
            f"  failure |obj_rel_base_y| mean={failure_rel_y.mean():.4f} p90={np.percentile(failure_rel_y, 90):.4f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    category_id = CATEGORY_NAME_TO_ID[args.category]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants, rank_env_to_variant = load_assignment(args.generation_assignment)
    stats = load_attempt_stats(args.run_dir, category_id, variants, rank_env_to_variant)
    success_rows = load_stored_states(args.run_dir, category_id, variants, rank_env_to_variant, success=True)
    failure_rows = load_stored_states(args.run_dir, category_id, variants, rank_env_to_variant, success=False)

    write_csv(args.output_dir / "object_success_rates.csv", stats["per_object_rows"])
    write_csv(args.output_dir / "variant_success_rates.csv", stats["per_variant_rows"])

    plot_object_success_rates(args.output_dir / "01_object_success_rates.png", stats["per_object_rows"])
    plot_variant_histogram(args.output_dir / "02_variant_success_rate_histogram.png", stats["per_variant_rows"])
    plot_variant_feature_scatter(args.output_dir / "03_variant_success_vs_features.png", stats["per_variant_rows"])
    plot_state_hexbin(args.output_dir / "04_success_vs_failure_start_geometry.png", success_rows, failure_rows)
    plot_failure_share_heatmap(args.output_dir / "05_failure_share_heatmap.png", success_rows, failure_rows)

    if args.compare_run_dir is not None:
        compare_success_rows = load_stored_states(
            args.compare_run_dir,
            category_id,
            variants,
            rank_env_to_variant,
            success=True,
            correct_only=args.compare_correct_side_only,
        )
        write_csv(args.output_dir / "compare_success_states.csv", compare_success_rows[: min(len(compare_success_rows), 100000)])
        plot_compare_success_geometry(
            args.output_dir / "06_compare_success_geometry.png",
            success_rows,
            compare_success_rows,
        )

    write_summary(
        args.output_dir / "summary.txt",
        args.run_dir,
        stats,
        stats["per_object_rows"],
        stats["per_variant_rows"],
        success_rows,
        failure_rows,
    )


if __name__ == "__main__":
    main()
