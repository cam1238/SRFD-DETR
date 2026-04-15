"""
SRFD-DETR 验证 / 测试集评估，并打印论文用表格（参数量、速度、mAP 等）。

用法:
  python val.py --weights runs/train/exp/weights/best.pt --data dataset/data.yaml --split test
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from prettytable import PrettyTable

from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate SRFD-DETR (RT-DETR variant)")
    p.add_argument(
        "--weights",
        type=str,
        default=os.environ.get("RTDETR_WEIGHTS", ""),
        help="训练好的 .pt 权重路径",
    )
    p.add_argument(
        "--data",
        type=str,
        default=os.environ.get("RTDETR_DATA", ""),
        help="数据集 data.yaml（与 train.py 相同格式）",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="在 data.yaml 中选择的划分",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument(
        "--save-json",
        action="store_true",
        help="保存 COCO 风格 JSON 以便计算 COCO 指标",
    )
    p.add_argument("--project", type=str, default="runs/val")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument(
        "--device",
        type=str,
        default=os.environ.get("RTDETR_DEVICE", ""),
        help="例如 0；空则自动",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.weights:
        raise SystemExit(
            "请指定权重: python val.py --weights path/to/best.pt --data path/to/data.yaml\n"
            "或设置环境变量 RTDETR_WEIGHTS。"
        )
    if not args.data:
        raise SystemExit(
            "请指定数据: python val.py --weights ... --data path/to/data.yaml\n"
            "或设置环境变量 RTDETR_DATA。"
        )

    model = RTDETR(args.weights)
    val_kw = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
    )
    if args.device:
        val_kw["device"] = args.device

    result = model.val(**val_kw)

    if model.task != "detect":
        print("当前任务非 detect，跳过扩展表格。")
        raise SystemExit(0)

    length = result.box.p.size
    model_names = list(result.names.values())
    preprocess_time_per_image = result.speed["preprocess"]
    inference_time_per_image = result.speed["inference"]
    postprocess_time_per_image = result.speed["postprocess"]
    all_time_per_image = (
        preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
    )

    n_l, n_p, n_g, flops = model_info(model.model)

    print("-" * 20 + "论文上的数据以以下结果为准" + "-" * 20)

    model_info_table = PrettyTable()
    model_info_table.title = "Model Info"
    model_info_table.field_names = [
        "GFLOPs",
        "Parameters",
        "前处理时间/一张图",
        "推理时间/一张图",
        "后处理时间/一张图",
        "FPS(前处理+模型推理+后处理)",
        "FPS(推理)",
        "Model File Size",
    ]
    model_info_table.add_row(
        [
            f"{flops:.1f}",
            f"{n_p:,}",
            f"{preprocess_time_per_image / 1000:.6f}s",
            f"{inference_time_per_image / 1000:.6f}s",
            f"{postprocess_time_per_image / 1000:.6f}s",
            f"{1000 / all_time_per_image:.2f}",
            f"{1000 / inference_time_per_image:.2f}",
            f"{get_weight_size(args.weights)}MB",
        ]
    )
    print(model_info_table)

    model_metrice_table = PrettyTable()
    model_metrice_table.title = "Model Metrice"
    model_metrice_table.field_names = [
        "Class Name",
        "Precision",
        "Recall",
        "F1-Score",
        "mAP50",
        "mAP75",
        "mAP50-95",
    ]
    for idx in range(length):
        model_metrice_table.add_row(
            [
                model_names[idx],
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                f"{result.box.all_ap[idx, 5]:.4f}",
                f"{result.box.ap[idx]:.4f}",
            ]
        )
    model_metrice_table.add_row(
        [
            "all(平均数据)",
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(result.box.f1[:length]):.4f}",
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f"{np.mean(result.box.all_ap[:length, 5]):.4f}",
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
        ]
    )
    print(model_metrice_table)

    with open(result.save_dir / "paper_data.txt", "w+", errors="ignore", encoding="utf-8") as f:
        f.write(str(model_info_table))
        f.write("\n")
        f.write(str(model_metrice_table))

    print("-" * 20, f"结果已保存至 {result.save_dir}/paper_data.txt", "-" * 20)
