#!/usr/bin/env python3
"""Show what is really on the USB bus, decoded by interface class.

Device-level class is usually 0 ("see the interfaces"), so a device with no
name tells you nothing on its own. The interface class is what identifies it,
and it is also what decides whether a serial driver will ever attach.

    ./usbcheck.py           # one-shot listing
    ./usbcheck.py --watch   # live: plug/unplug and watch devices arrive and leave
"""

import plistlib
import subprocess
import sys
import time

# Interface classes. The SO-101 needs one of the SERIAL ones to appear.
IFACE_CLASS = {
    1: "Audio",
    2: "CDC-Comm",
    3: "HID",
    6: "Image/PTP",
    7: "Printer",
    8: "Mass Storage",
    9: "Hub",
    10: "CDC-Data",
    11: "SmartCard",
    14: "Video/UVC",
    224: "Wireless",
    255: "Vendor-Specific",
}

# A serial port shows up for CDC-ACM (2/2) or a vendor-specific bridge such as
# CH340 or CP210x (255). CDC subclass 6 is Ethernet, NOT a serial port -- that
# is what a USB network adapter looks like, and it is easy to misread as one.
#
# Judge this per device, not per interface: a CDC-Data interface (class 10) is
# the bulk endpoint of whatever control interface sits beside it, so it is
# serial next to CDC-ACM and ethernet next to CDC-ECM.
def device_is_serial(ifaces: list) -> bool:
    if any(cls == 2 and sub == 6 for cls, sub in ifaces):
        return False  # ECM control interface -> this is a network adapter
    return any(cls == 2 or cls == 10 or cls == 255 for cls, sub in ifaces)


def iface_label(cls: int, sub: int) -> str:
    if cls == 2 and sub == 6:
        return "CDC-Ethernet"
    return IFACE_CLASS.get(cls, f"class {cls}")


def ioreg(*args: str):
    out = subprocess.run(["ioreg", "-a", "-w0", *args], capture_output=True).stdout
    return plistlib.loads(out) if out.strip() else []


def devices() -> dict:
    """locationID -> {vid, pid, name, ifaces:[(class, sub)]}"""
    root = ioreg("-p", "IOUSB", "-l")
    found = {}

    def walk(node):
        if isinstance(node, dict):
            if "idVendor" in node:
                found[node.get("locationID", 0)] = {
                    "vid": node.get("idVendor", 0),
                    "pid": node.get("idProduct", 0),
                    "name": node.get("USB Product Name") or "(unnamed)",
                    "ifaces": [],
                }
            for child in node.get("IORegistryEntryChildren", []) or []:
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(root)

    for iface in ioreg("-r", "-c", "IOUSBHostInterface", "-l"):
        dev = found.get(iface.get("locationID", 0))
        if dev is not None:
            dev["ifaces"].append(
                (iface.get("bInterfaceClass"), iface.get("bInterfaceSubClass"))
            )
    return found


def serial_ports() -> list:
    out = subprocess.run(
        ["bash", "-c", "ls /dev/tty.usbmodem* /dev/tty.usbserial* /dev/tty.wchusbserial* 2>/dev/null"],
        capture_output=True, text=True,
    ).stdout
    return [p for p in out.split() if p]


def describe(dev: dict) -> str:
    parts = [iface_label(cls, sub) for cls, sub in dev["ifaces"]]
    text = ", ".join(parts) or "(no interfaces reported)"
    if device_is_serial(dev["ifaces"]):
        text += "   <-- SERIAL DEVICE"
    return text


def report() -> None:
    devs = devices()
    print(f"{len(devs)} USB devices\n")
    for loc in sorted(devs):
        d = devs[loc]
        print(f"  0x{loc:08x}  {d['vid']:#06x}:{d['pid']:#06x}  {d['name']}")
        print(f"              {describe(d)}")

    candidates = [d for d in devs.values() if device_is_serial(d["ifaces"])]
    ports = serial_ports()

    print()
    if ports:
        print("SERIAL PORTS:")
        for p in ports:
            print(f"  {p}")
        print("\nReady. Use these with lerobot.")
    elif candidates:
        print("A serial-capable device is present but no /dev port appeared.")
        print("That points at a missing driver rather than a cable.")
    else:
        print("NO serial device on the bus -- the arm is not enumerating.")
        print("Nothing in software can fix this. In likelihood order:")
        print("  1. Charge-only cable        -> swap for a known data cable")
        print("  2. Arm power off            -> check supply and switch")
        print("  3. Cable not fully seated   -> reseat both ends, listen for click")
        print("  4. Dead hub port            -> try another port on the hub")


def watch() -> None:
    print("Watching USB. Plug or unplug the arm. Ctrl-C to stop.\n")
    prev, prev_ports = devices(), set(serial_ports())
    while True:
        time.sleep(1)
        cur, cur_ports = devices(), set(serial_ports())
        for loc in set(cur) - set(prev):
            d = cur[loc]
            print(f"+ ARRIVED  {d['vid']:#06x}:{d['pid']:#06x} {d['name']} -- {describe(d)}")
        for loc in set(prev) - set(cur):
            d = prev[loc]
            print(f"- REMOVED  {d['vid']:#06x}:{d['pid']:#06x} {d['name']}")
        for p in cur_ports - prev_ports:
            print(f"*** SERIAL PORT READY: {p}")
        for p in prev_ports - cur_ports:
            print(f"*** serial port gone: {p}")
        prev, prev_ports = cur, cur_ports


if __name__ == "__main__":
    try:
        watch() if "--watch" in sys.argv else report()
    except KeyboardInterrupt:
        print("\nstopped")
