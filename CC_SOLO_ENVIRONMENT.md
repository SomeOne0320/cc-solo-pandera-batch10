# Fixed local pandas environment

This snapshot is based on the published Pandera v0.36.0 tag, upstream SHA2a5d1d6997a2c2faa0d6c8d936d3a8d656d0d741. Use Python3.11/LinuxARM64 in the standard model image with only the local pandas backend. All actual runtime/testing packages are pinned in requirements-cc-solo.txt. No external database, cloud service or other validation backend is needed.

```bash
curl --connect-timeout 20 --max-time 180 -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/pandera-get-pip.py
python3 /tmp/pandera-get-pip.py --user --break-system-packages pip==24.3.1
python3 -m pip install --user --break-system-packages -r requirements-cc-solo.txt
SETUPTOOLS_SCM_PRETEND_VERSION=0.36.0 python3 -m pip install --user --break-system-packages --no-build-isolation --no-deps -e .
python3 -m pytest -q tests/pandas/test_drop_invalid_rows_failure_limit.py
```

The build command generates the ignored pandera/_version.py, so it is required even when importing directly from this checkout. The fixed build-version override keeps metadata reproducible in the shallow evaluation checkout. The initial release passes the selected existing failure-limit tests on pandas2.3.3/numpy1.26.4. Use relevant additional local tests for changed behavior. YAML support is available via the pinned PyYAML package, without installing Frictionless or remote backends.

Read AGENTS.md for architecture, compatibility, typing and code style. Its all-extras developer setup is unnecessary for this pandas-only task; do not install Spark/Ray/other backend services. Use python3 -m ruff for changed files (79-column formatting) and preserve old behavior except the specifically requested change. An unavailable optional-backend integration is not a tested result. Do not create an upstream issue/PR or publish a package release.
