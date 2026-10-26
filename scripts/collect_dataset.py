"""Dataset Collection Utility for Texas Instruments IWR6843ISK-ODS mmWave Radar.

Streamlines collecting labeled posture and fall trials into the project dataset:
- Direct CLI command mode:
    python scripts/collect_dataset.py --subject person1 --activity falling --duration 10
- Guided interactive loop mode:
    python scripts/collect_dataset.py --interactive

Output structure matches the training pipeline:
    dataset/my_posture_dataset/<subject>/<activity>/T<trial>_<timestamp>/targets.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.radar.data_collector import RadarDataCollector
from src.radar.serial_acquisition import RadarSerialAcquisition

VALID_ACTIVITIES = ["falling", "lying", "standing", "sitting", "walking", "upright"]


def run_collection(
    cli_port: str = "COM11",
    data_port: str = "COM14",
    subject: str = "person1",
    activity: str = "falling",
    trial: int | str = 1,
    duration: float = 10.0,
    cfg_path: str | Path = "configs/ODS_6m_posture_tuned.cfg",
    out_dir: str | Path = "dataset/my_posture_dataset",
    notes: str = "",
) -> Path | None:
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = PROJECT_ROOT / cfg_file

    if not cfg_file.exists():
        print(f"[Error] Configuration file not found: {cfg_file}")
        return None

    # Normalize COM port format
    cli_port = cli_port if cli_port.upper().startswith("COM") else f"COM{cli_port}"
    data_port = data_port if data_port.upper().startswith("COM") else f"COM{data_port}"

    # Format trial display
    t_str = str(trial).strip()
    t_display = t_str.upper() if t_str.upper().startswith("T") else f"T{int(t_str):02d}" if t_str.isdigit() else f"T_{t_str}"

    print("\n" + "=" * 65)
    print(" Texas Instruments mmWave Radar Dataset Collection")
    print("=" * 65)
    print(f" Subject    : {subject}")
    print(f" Activity   : {activity.upper()}")
    print(f" Trial      : {t_display}")
    print(f" Duration   : {duration:.1f} seconds" if duration > 0 else " Duration   : Manual (press Ctrl+C or Enter to stop)")
    print(f" Radar CLI  : {cli_port} (Baud 115200)")
    print(f" Radar DATA : {data_port} (Baud 921600)")
    print(f" Config     : {cfg_file.name}")
    print("=" * 65)

    acq = RadarSerialAcquisition(
        cli_port=cli_port,
        data_port=data_port,
        cli_baud=115200,
        data_baud=921600,
        timeout=0.8,
    )

    collector = RadarDataCollector(
        output_base_dir=out_dir,
        subject_id=subject,
        activity_label=activity.lower(),
        trial_number=trial,
        notes=notes,
        cfg_path=str(cfg_file),
    )

    try:
        print("\n[1/3] Transmitting chirp configuration to radar CLI...")
        acq.send_configuration(cfg_file)
        print("[1/3] Chirp configuration accepted! Sensor initialized.")

        print("\n[2/3] Connecting to Radar Data port...")
        acq.connect_data_port()
        print("[2/3] Connected to Data UART stream.")

        # 3-second audible / visual countdown
        print("\nGet ready! Recording starts in:")
        for countdown in range(3, 0, -1):
            print(f"   >>> {countdown} <<<")
            time.sleep(1.0)
        print("   >>> RECORDING NOW! <<<\n")

        collector.start()
        start_time = time.time()
        last_display = 0.0

        while True:
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                break

            try:
                frame = acq.read_frame()
                collector.record_frame(frame)
            except TimeoutError:
                continue
            except Exception as e:
                print(f"\n[Warning] Frame capture glitch: {e}")
                continue

            # Status line update every 0.25 seconds
            if time.time() - last_display > 0.25:
                last_display = time.time()
                num_t = len(frame.targets)
                num_p = len(frame.points)
                t_info = ""
                if num_t > 0:
                    t = frame.targets[0]
                    t_info = f" | Target0: Z={t.pos_z:.2f}m, Vz={t.vel_z:.2f}m/s"

                dur_str = f"{elapsed:4.1f}s / {duration:4.1f}s" if duration > 0 else f"{elapsed:4.1f}s"
                sys.stdout.write(f"\r[REC] Time: {dur_str} | Frames: {collector.frame_count:4d} | Pts: {num_p:3d}{t_info}   ")
                sys.stdout.flush()

        print(f"\n\n[3/3] Session finished. Saving files...")
        saved_dir = collector.stop()
        print(f"[Done] Trial saved successfully to:\n       {saved_dir}")
        return saved_dir

    except KeyboardInterrupt:
        print("\n\n[Info] Capture stopped by user.")
        saved_dir = collector.stop()
        return saved_dir
    except Exception as e:
        print(f"\n[Error] Radar connection failed: {e}")
        collector.stop()
        return None
    finally:
        acq.close()
        print("[Info] Radar ports closed and sensor stopped cleanly.\n")


def interactive_mode(cli_port: str = "COM11", data_port: str = "COM14"):
    print("\n" + "#" * 65)
    print(" TEXAS INSTRUMENTS MMWAVE RADAR - GUIDED DATA COLLECTION")
    print("#" * 65)

    cli = input(f"Enter Radar CLI COM port [{cli_port}]: ").strip() or cli_port
    data = input(f"Enter Radar DATA COM port [{data_port}]: ").strip() or data_port

    while True:
        print("\nSelect Subject:")
        print("  1) person1    2) person2    3) person3")
        print("  4) person4    5) person5    6) person6")
        sub_choice = input("Enter choice (1-6) or custom name [1]: ").strip() or "1"
        subject = f"person{sub_choice}" if sub_choice.isdigit() and 1 <= int(sub_choice) <= 6 else sub_choice

        print("\nSelect Activity / Posture:")
        print("  1) falling  (Rapid downward collapse)")
        print("  2) lying    (Person flat on floor / bed)")
        print("  3) standing (Upright stationary stance)")
        print("  4) sitting  (Seated on chair / floor)")
        print("  5) walking  (Normal locomotion)")
        act_map = {
            "1": "falling",
            "2": "lying",
            "3": "standing",
            "4": "sitting",
            "5": "walking",
        }
        act_choice = input("Enter choice (1-5) [1]: ").strip() or "1"
        activity = act_map.get(act_choice, "falling")

        trial_input = input("Enter trial number (e.g. 1, 2, 3) [1]: ").strip() or "1"
        dur_input = input("Enter duration in seconds [10]: ").strip() or "10"
        try:
            duration = float(dur_input)
        except ValueError:
            duration = 10.0

        run_collection(
            cli_port=cli,
            data_port=data,
            subject=subject,
            activity=activity,
            trial=trial_input,
            duration=duration,
        )

        again = input("Do you want to record another trial? (y/n) [y]: ").strip().lower()
        if again in ("n", "no", "q", "quit"):
            print("Exiting data collection. Happy model training!\n")
            break


def main():
    parser = argparse.ArgumentParser(description="mmWave Radar Dataset Collection Utility")
    parser.add_argument("--cli", default="COM11", help="Radar CLI COM port (default: COM11)")
    parser.add_argument("--data", default="COM14", help="Radar Data COM port (default: COM14)")
    parser.add_argument("--subject", default="person1", help="Subject name (e.g. person1, person2)")
    parser.add_argument("--activity", default="falling", choices=VALID_ACTIVITIES, help="Activity posture label")
    parser.add_argument("--trial", default="1", help="Trial number or identifier (e.g. 1, 2, 3, T01, T02)")
    parser.add_argument("--duration", type=float, default=10.0, help="Trial recording duration in seconds (default: 10)")
    parser.add_argument("--cfg", default="configs/ODS_6m_posture_tuned.cfg", help="Chirp configuration path")
    parser.add_argument("--out-dir", default="dataset/my_posture_dataset", help="Dataset destination folder")
    parser.add_argument("--notes", default="", help="Optional trial notes")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive guided collection menu")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(cli_port=args.cli, data_port=args.data)
    else:
        run_collection(
            cli_port=args.cli,
            data_port=args.data,
            subject=args.subject,
            activity=args.activity,
            trial=args.trial,
            duration=args.duration,
            cfg_path=args.cfg,
            out_dir=args.out_dir,
            notes=args.notes,
        )


if __name__ == "__main__":
    main()
