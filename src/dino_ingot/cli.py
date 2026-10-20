"""Desktop workflows. Run `ingot --help` for command syntax."""

import argparse
import copy
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

from .camera import (
    clear_failure,
    open_camera,
    quit_requested,
    read_frame,
    require_gui,
    show_inspection,
)
from .config import Config, load_config, sha256_file
from .dataset import load_records, save_sample, validate_item_id
from .vision import extract_region, load_background, save_image

DEFAULT_WEIGHTS = "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def make_extractor(args, config: Config):
    from .features import DinoExtractor

    return DinoExtractor(args.repo, args.weights, config.features.input_size, args.device)


def calibrate(args) -> None:
    require_gui()
    config = load_config(args.config)
    if args.output.exists():
        raise ValueError("調整の出力先が既に存在します。新しいディレクトリを指定してください")
    window = "Ingot calibration (s=save, q=cancel)"
    with open_camera(config.camera) as capture:
        print("インゴットを取り除き、検査範囲をドラッグ選択してEnter。Escで中止。", flush=True)
        frame = read_frame(capture, config.camera)
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        if roi[2] == 0 or roi[3] == 0:
            print("調整を中止しました")
            return
        config.extraction.roi = [int(v) for v in roi]
        frames = [read_frame(capture, config.camera) for _ in range(15)]
        background = np.median(np.stack(frames), axis=0).astype(np.uint8)
        print("背景を取得しました。インゴットを置いて閾値・最小面積を調整し、sで保存。", flush=True)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("threshold", window, config.extraction.threshold, 255, lambda _: None)
        cv2.createTrackbar(
            "min_area",
            window,
            config.extraction.min_area,
            max(config.extraction.min_area, roi[2] * roi[3]),
            lambda _: None,
        )
        while True:
            frame = read_frame(capture, config.camera)
            config.extraction.threshold = cv2.getTrackbarPos("threshold", window)
            config.extraction.min_area = max(1, cv2.getTrackbarPos("min_area", window))
            region = extract_region(frame, background, config.extraction)
            show_inspection(
                window, frame, region, config.extraction.roi, f"{region.status} | s=save q=cancel"
            )
            mask = region.mask
            scale = min(1.0, 800 / mask.shape[1], 600 / mask.shape[0])
            cv2.imshow("Difference mask", cv2.resize(mask, None, fx=scale, fy=scale))
            key = cv2.waitKey(1) & 0xFF
            if quit_requested(window, key):
                return
            if key == ord("s"):
                config.validate()
                args.output.mkdir(parents=True)
                save_image(args.output / "background.png", background)
                config.background = "background.png"
                config.background_sha256 = sha256_file(args.output / "background.png")
                config.save(args.output / "config.json")
                print(f"保存しました: {args.output / 'config.json'}")
                return


def capture_samples(args) -> None:
    require_gui()
    config = load_config(args.config)
    background = load_background(config)
    validate_item_id(args.item_id)
    for record in load_records(args.data):
        if record["item_id"] == args.item_id and record["label"] != args.label:
            raise ValueError("同じ個体IDに異なるラベルを付けられません")
        if record["config_id"] != config.identity():
            raise ValueError(
                "撮影データと設定が異なります。別のデータディレクトリを指定してください"
            )
    window = "Ingot capture"
    count = 0
    print(f"個体={args.item_id} ラベル={args.label}。Spaceで撮影、qで終了。", flush=True)
    with open_camera(config.camera) as camera:
        try:
            while True:
                frame = read_frame(camera, config.camera)
                region = extract_region(frame, background, config.extraction)
                text = (
                    f"{region.status} | {args.item_id} {args.label} | "
                    f"saved={count} SPACE=save q=quit"
                )
                show_inspection(window, frame, region, config.extraction.roi, text)
                key = cv2.waitKey(1) & 0xFF
                if quit_requested(window, key):
                    break
                if key == ord(" "):
                    if region.status != "OK":
                        print(f"保存しません: {region.status} {region.reason}")
                        continue
                    target = save_sample(args.data, frame, region, config, args.item_id, args.label)
                    count += 1
                    print(f"保存しました: {target}", flush=True)
        except Exception:
            clear_failure(window)
            raise


def train_command(args) -> None:
    from .training import train

    config = load_config(args.config)
    # Check user inputs before the potentially expensive model load.
    from .dataset import split_items

    if args.output.exists():
        raise ValueError("出力先が既に存在します。別のモデルディレクトリを指定してください")
    load_background(config)
    split_items(load_records(args.data), args.seed)
    report = train(args.data, args.output, config, make_extractor(args, config), args.seed)
    print(
        json.dumps(
            {"model": str(args.output), "test": report["test"]}, ensure_ascii=False, indent=2
        )
    )


def infer(args) -> None:
    from .training import load_bundle, verify_extractor
    from .ur3 import WristController, load_ur3_config

    ur3_path = getattr(args, "ur3_config", None)
    dry_run = getattr(args, "ur3_dry_run", False)
    if dry_run and ur3_path is None:
        raise ValueError("--ur3-dry-run には --ur3-config が必要です")
    ur3_config = load_ur3_config(ur3_path) if ur3_path is not None else None
    require_gui()
    config, background, model, metadata = load_bundle(args.model)
    runtime = copy.deepcopy(config)
    if args.camera is not None:
        # /dev/video indices may change after reboot; geometry and calibration stay fixed.
        runtime.camera.source = int(args.camera) if args.camera.isdecimal() else args.camera
        runtime.validate()
    extractor = make_extractor(args, config)
    verify_extractor(metadata, extractor)
    window = "Ingot inference"
    print("qで終了。異常スコアは確率ではありません。", flush=True)
    with open_camera(runtime.camera) as camera:
        controller = None
        try:
            controller = (
                WristController(ur3_config, dry_run=dry_run) if ur3_config else nullcontext()
            )
            with controller as robot:
                while True:
                    if robot is not None:
                        robot.check()
                    frame_started = time.monotonic()
                    start = time.perf_counter()
                    frame = read_frame(camera, runtime.camera)
                    region = extract_region(frame, background, config.extraction)
                    verdict = region.status
                    text = verdict
                    if region.status == "OK":
                        vector = extractor.extract(region.crop)[None]
                        if vector.shape[1] != metadata["feature_dimension"]:
                            raise ValueError("学習時と特徴量数が異なります")
                        label = int(model.predict(vector)[0])
                        if label not in (0, 1):
                            raise ValueError("推論ラベルは正常=0または異常=1である必要があります")
                        score = float(model.decision_function(vector)[0])
                        if not np.isfinite(score):
                            raise ValueError("異常スコアが有限値ではありません")
                        verdict = "ANOMALY" if label == 1 else "NORMAL"
                        text = f"{verdict} | score={score:+.3f}"
                    elapsed = (time.perf_counter() - start) * 1000
                    robot_text = ""
                    if robot is not None:
                        robot.publish(verdict, frame_started)
                        robot_text = robot.status()
                    show_inspection(
                        window,
                        frame,
                        region,
                        config.extraction.roi,
                        f"{text} | {elapsed:.0f}ms | q=quit",
                        **({"extra_text": robot_text} if robot is not None else {}),
                    )
                    if quit_requested(window, cv2.waitKey(1) & 0xFF):
                        break
        except Exception:
            detail = ""
            if ur3_config is not None:
                confirmed = controller is not None and controller.stopped_confirmed
                detail = f"UR3 STOP {'CONFIRMED' if confirmed else 'UNCONFIRMED'} - see terminal"
            clear_failure(window, **({"detail": detail} if detail else {}))
            raise


def doctor(args) -> int:
    from .features import verify_repository

    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append(True)
            print(f"[OK] {name}: {detail}")
        except Exception as error:
            checks.append(False)
            print(f"[FAIL] {name}: {error}")

    config = load_config(args.config)
    check("OpenCV / NumPy", lambda: f"{cv2.__version__} / {np.__version__}")
    check("GUI", lambda: (require_gui(), "GUIビルド・画面環境変数を確認（表示自体は未検証）")[1])

    def torch_check():
        import torch

        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA利用不可。JetPack対応のPyTorchを導入してください")
            torch.ones(1, device="cuda").sum().item()
            return f"{torch.__version__}, {torch.cuda.get_device_name(0)}"
        return f"{torch.__version__}, CPU指定"

    check("PyTorch / device", torch_check)
    check("DINOv3 repository", lambda: verify_repository(args.repo))
    check("DINOv3 weights SHA256", lambda: sha256_file(args.weights))

    def camera_check():
        with open_camera(config.camera) as camera:
            return str(read_frame(camera, config.camera).shape)

    check("USB camera", camera_check)
    if config.background:
        check("background", lambda: str(load_background(config).shape))
    if args.load_model and all(checks):
        check(
            "DINOv3 forward",
            lambda: str(
                make_extractor(args, config).extract(np.zeros((256, 256, 3), dtype=np.uint8)).shape
            ),
        )
    return 0 if all(checks) else 1


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="DINOv3 + SVM によるインゴット検査")
    commands = cli.add_subparsers(dest="command", required=True)

    def config_arg(command, required=True):
        command.add_argument(
            "--config",
            type=Path,
            required=required,
            default=None if required else Path("config.example.json"),
        )

    def model_args(command):
        command.add_argument("--repo", type=Path, default=Path("vendor/dinov3"))
        command.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS))
        command.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    command = commands.add_parser("doctor", help="実行環境を診断")
    config_arg(command, False)
    model_args(command)
    command.add_argument("--load-model", action="store_true", help="DINOv3の実際の順伝播も確認")
    command.set_defaults(run=doctor)
    command = commands.add_parser("calibrate", help="検査範囲・背景・抽出条件を保存")
    config_arg(command, False)
    command.add_argument("--output", type=Path, default=Path("calibration/default"))
    command.set_defaults(run=calibrate)
    command = commands.add_parser("capture", help="個体IDとラベルを指定して写真を収集")
    config_arg(command)
    command.add_argument("--data", type=Path, default=Path("data/ingots"))
    command.add_argument("--item-id", required=True)
    command.add_argument("--label", choices=["normal", "anomaly"], required=True)
    command.set_defaults(run=capture_samples)
    command = commands.add_parser("train", help="DINOv3特徴からSVMを学習・評価")
    config_arg(command)
    model_args(command)
    command.add_argument("--data", type=Path, default=Path("data/ingots"))
    command.add_argument("--output", type=Path, default=Path("models/ingot-v1"))
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(run=train_command)
    command = commands.add_parser("infer", help="USBカメラ映像を判定")
    model_args(command)
    command.add_argument("--model", type=Path, default=Path("models/ingot-v1"))
    command.add_argument("--camera", help="再起動でカメラ番号が変わった場合のみ番号・パスを変更")
    command.add_argument("--ur3-config", type=Path, help="UR3連携の設定JSON（指定時のみ接続）")
    command.add_argument("--ur3-dry-run", action="store_true", help="実機接続せずUR3指令を確認")
    command.set_defaults(run=infer)
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args) or 0
    except KeyboardInterrupt:
        print("中止しました", file=sys.stderr)
        return 130
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        ImportError,
        subprocess.SubprocessError,
        cv2.error,
    ) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
