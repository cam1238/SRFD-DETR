"""
SRFD-DETR 检测端训练入口。

用法示例:
  python train.py --data dataset/data.yaml
  python train.py --data dataset/data.yaml --cfg ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml --epochs 300 --batch 4

环境变量（可选，与命令行互斥时以命令行为准）:
  RTDETR_DATA   数据集 yaml 路径
  RTDETR_DEVICE 例如 0 或 0,1
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from ultralytics import RTDETR


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SRFD-DETR (RT-DETR variant)")
    p.add_argument(
        "--data",
        type=str,
        default=os.environ.get("RTDETR_DATA", ""),
        help="数据集 data.yaml 路径（YOLO 格式；含 train/val/test 与 nc 等）",
    )
    p.add_argument(
        "--cfg",
        type=str,
        default="ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml",
        help="模型结构 yaml；主方法解耦见 rtdetr-r18.yaml，尺度嵌入基线见 rtdetr-r18-scale-embed.yaml",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="",
        help="可选：预训练权重 .pt 路径，空则不加载",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=4, help="Windows 卡死时可改为 0")
    p.add_argument("--cache", action="store_true", help="是否缓存数据到内存")
    p.add_argument(
        "--device",
        type=str,
        default=os.environ.get("RTDETR_DEVICE", ""),
        help="例如 0 或 0,1；空则由 Ultralytics 自动选择",
    )
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="srfd-detr-exp")
    p.add_argument("--resume", type=str, default="", help="从 last.pt 等检查点续训")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.data:
        raise SystemExit(
            "请指定数据集: python train.py --data /path/to/data.yaml\n"
            "或设置环境变量 RTDETR_DATA。"
        )

    model = RTDETR(args.cfg)
    if args.weights:
        model.load(args.weights)

    train_kw = dict(
        data=args.data,
        cache=args.cache,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )
    if args.device:
        train_kw["device"] = args.device
    if args.resume:
        train_kw["resume"] = args.resume

    model.train(**train_kw)
