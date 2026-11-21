# Contributing

Thanks for your interest. This project is small and the maintainers review
every change.

## Setup

```bash
git clone https://github.com/manascb1344/visdrone-surveillance.git
cd visdrone-surveillance
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Python 3.8 or newer is required. FFmpeg is needed for streaming features.

## Before you open a pull request

1. Run the test suite: `python -m pytest tests/ -q`. All tests must pass.
2. Keep the change focused. One concern per pull request.
3. Match the existing code style: relative imports inside the package,
   no `sys.path` hacks, docstrings on public functions.
4. Add or update tests for any behavior change.

## What goes where

- `main/visdrone_surveillance/` is the supported package. All production
  code goes here.
- `research/` holds prototypes, notebooks, and experiment artifacts. It is
  not tested or packaged.
- `tests/` pins package behavior. Offline, no GPU, no network, no dataset.
- `docs/` holds architecture and status documentation.

## Reporting issues

Open an issue with the steps to reproduce, the expected behavior, and the
actual behavior. Include your Python version and operating system.

## License

By contributing you agree that your contributions are licensed under the
MIT license. See `LICENSE`.