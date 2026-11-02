"""
cli.py — Master Interactive Control Panel for Antigravity Quant V3+.

Launch with:
    .venv\\Scripts\\python.exe cli.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print(r"""
========================================================================
       ___          __  _                           _ _           
      / _ \ _   _  / _|| | _____      __  /\  /\___(_) | ___  ___ 
     / /_\/ | | | | |_ | |/ _ \ \ /\ / / / /_/ / _ \ | |/ _ \/ __|
    / /_\\| |_| | |  _|| | (_) \ V  V / / __  /  __/ | |  __/\__ \
    \____/ \__,_| |_|  |_|\___/ \_/\_/  \/ /_/ \___|_|_|\___||___/
    
              ANTIGRAVITY QUANT V3+ — MASTER CONTROL PANEL
========================================================================
""")


def main_menu():
    py_exec = sys.executable

    while True:
        clear_screen()
        print_banner()
        print("  [1] Audit a Single Strategy CSV      (Generates Factsheet & JSON)")
        print("  [2] Run ML Meta-Labeling Optimizer   (Finds Best Hurdle & Compares Curves)")
        print("  [3] Batch Process Inbox -> Outbox    (Auto-creates client ZIP packages)")
        print("  [4] Recompile Master Leaderboard     (Updates reports/league_table.html)")
        print("  [5] Launch Web Dashboard             (Opens Flask server on port 5000)")
        print("  [6] Start Autonomous Paper Trader    (20-min cycle intraday trading bot)")
        print("  [7] Run Intraday Session Simulator   (Offline AAPL simulation cycle)")
        print("  [8] Run Regression Test Suite        (Executes 120 automated tests)")
        print("  [9] Open Web Reports in Browser      (Factsheet / Leaderboard)")
        print("  [0] Exit")
        print("=" * 72)

        choice = input("  Select an action [0-9]: ").strip()

        if choice == "0":
            print("\nExiting Antigravity Quant. Good trading!\n")
            break

        elif choice == "1":
            print("\n--- AUDIT STRATEGY CSV ---")
            default_csv = "reports/client_strategy/client_trades.csv"
            csv_path = input(f"Path to trades CSV (default: {default_csv}): ").strip() or default_csv
            out_dir = input("Output directory (default: reports/client_audit): ").strip() or "reports/client_audit"
            cmd = [py_exec, "audit_strategy.py", "--csv", csv_path, "--out", out_dir]
            subprocess.run(cmd)
            input("\nPress Enter to return to menu...")

        elif choice == "2":
            print("\n--- ML META-LABELING OPTIMIZER ---")
            default_csv = "reports/client_strategy/client_trades.csv"
            csv_path = input(f"Path to trades CSV (default: {default_csv}): ").strip() or default_csv
            out_dir = input("Output directory (default: reports/meta_labeled): ").strip() or "reports/meta_labeled"
            cmd = [py_exec, "meta_label_optimizer.py", "--csv", csv_path, "--out", out_dir]
            subprocess.run(cmd)
            input("\nPress Enter to return to menu...")

        elif choice == "3":
            print("\n--- BATCH PROCESS INBOX -> OUTBOX ---")
            cmd = [py_exec, "batch_processor.py"]
            subprocess.run(cmd)
            input("\nPress Enter to return to menu...")

        elif choice == "4":
            print("\n--- RECOMPILE MASTER STRATEGY LEADERBOARD ---")
            cmd = [py_exec, "generate_league_table.py"]
            subprocess.run(cmd)
            input("\nPress Enter to return to menu...")

        elif choice == "5":
            print("\n--- LAUNCHING WEB DASHBOARD ---")
            print("Access in browser at: http://localhost:5000")
            print("Press Ctrl+C to stop server.\n")
            try:
                subprocess.run([py_exec, "app.py"])
            except KeyboardInterrupt:
                print("\nServer stopped.")
            input("\nPress Enter to return to menu...")

        elif choice == "6":
            print("\n--- LAUNCHING AUTONOMOUS PAPER TRADER ---")
            print("Press Ctrl+C to stop paper trader.\n")
            try:
                subprocess.run([py_exec, "auto_trader.py"])
            except KeyboardInterrupt:
                print("\nPaper trader stopped.")
            input("\nPress Enter to return to menu...")

        elif choice == "7":
            print("\n--- INTRADAY SESSION SIMULATION ---")
            cmd = [py_exec, "simulate_session.py", "--ticker", "AAPL"]
            subprocess.run(cmd)
            input("\nPress Enter to return to menu...")

        elif choice == "8":
            print("\n--- RUNNING 120-TEST REGRESSION SUITE ---")
            cmd = [py_exec, "-m", "pytest", "-q"]
            subprocess.run(cmd)
            input("\nPress Enter to return to menu...")

        elif choice == "9":
            print("\nWhich report would you like to open in browser?")
            print("  [a] Master Strategy Leaderboard (reports/league_table.html)")
            print("  [b] ML Meta-Labeled Factsheet (reports/meta_labeled/factsheet.html)")
            print("  [c] Base Strategy Factsheet (reports/client_strategy/factsheet.html)")
            sub_choice = input("Select [a/b/c]: ").strip().lower()

            target = None
            if sub_choice == "a":
                target = "reports/league_table.html"
            elif sub_choice == "b":
                target = "reports/meta_labeled/factsheet.html"
            elif sub_choice == "c":
                target = "reports/client_strategy/factsheet.html"

            if target and Path(target).is_file():
                if os.name == "nt":
                    os.system(f'start "" "{target}"')
                else:
                    os.system(f'xdg-open "{target}"')
                print(f"Opened: {target}")
            else:
                print("Report file not found. Run generation first.")
            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main_menu()
