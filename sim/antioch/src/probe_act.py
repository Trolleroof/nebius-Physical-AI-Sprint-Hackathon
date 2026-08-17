"""Report whether the Isaac runtime can load the ACT stack."""

import importlib.util

import antioch


def main() -> None:
    antioch.boot()
    for package in ("torch", "lerobot"):
        spec = importlib.util.find_spec(package)
        print(f"{package}: {spec.origin if spec else 'MISSING'}")


if __name__ == "__main__":
    main()
