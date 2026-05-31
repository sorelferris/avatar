#!/usr/bin/env python3
"""Minimal Joycon test script."""

from src.joycon import JoyconManager


def main():
    manager = JoyconManager()

    def on_button(side, key, pressed):
        print(f"{side.value}: {key} = {pressed}")

    manager.on_button = on_button

    count = manager.scan()
    print(f"Found {count} Joycon(s)")

    if count > 0:
        manager.start()
        print("Reading events for 10 seconds...")
        import time

        time.sleep(10)
        manager.stop()
        print("Done")
    else:
        print("No Joycons found. Make sure your Joycon is paired via Bluetooth.")


if __name__ == "__main__":
    main()
