#!/usr/bin/env python3
"""Local, serial supervisor for the version-1 evolution benchmark.

Only the Rust executable owns rendering, seeded evolution, and numerical checks.
This supervisor rejects incomplete work and failed replay, and preserves evidence.
A baseline is a reference series, not a campaign freeze or a speedup claim.

Power of Ten deviations: rules 3/8/9, Python JSON, argparse and unittest allocate
and dispatch through framework callbacks. Inputs, file counts, bytes, trials and
waits are capped; one child runs at a time, with scoped files and a POSIX lock.
Rule 10: stdlib warning-enabled unittest is the available validation; repository
CI configuration is outside this two-file slice. No GPU correctness is inferred
from the synthetic executable used by the tests. Future protocol changes must
retain these bounds and the timeout, integrity, accounting and replay tests.
"""

import argparse
import base64
import contextlib
import datetime
import fcntl
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / ".cache/evolution-bench"
PROJECTS = ("ff", "mona", "darwin")
MAX_CASES = 24
MAX_FILES = 512
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BINARY_BYTES = 512 * 1024 * 1024  # Release profile retains debug symbols.
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_WS_BYTES = 8 * 1024 * 1024
MAX_TRIALS = MAX_CASES * 20 * 3
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
ENV_KEYS = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTUP_TOOLCHAIN",
            "CARGO_BUILD_TARGET", "WGPU_BACKEND", "WGPU_POWER_PREF",
            "WGPU_ADAPTER_NAME", "VK_ICD_FILENAMES", "VK_DRIVER_FILES",
            "MESA_D3D12_DEFAULT_ADAPTER_NAME", "DISPLAY", "WAYLAND_DISPLAY")


class BenchError(Exception):
    """An invalid input, subprocess result, or benchmark acceptance condition."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise BenchError(message)


def integer(value, low, high, name):
    require(type(value) is int and low <= value <= high,
            f"{name} must be an integer in {low}..{high}")
    return value


def finite(value, low, high, name):
    require(type(value) in (int, float) and math.isfinite(value)
            and low <= value <= high, f"{name} must be finite in {low}..{high}")
    return value


def object_value(value, name):
    require(isinstance(value, dict) and bool(value), f"{name} must be a nonempty object")
    return value


def digest_value(value, name):
    require(isinstance(value, str) and SHA256.fullmatch(value), f"invalid {name}")
    return value


def sha_file(path, limit=MAX_FILE_BYTES):
    require(path.is_file() and not path.is_symlink(), f"missing or symlink file: {path}")
    size = path.stat().st_size
    require(size <= limit, f"file exceeds {limit} bytes: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for _ in range(limit // 65536 + 1):
            block = stream.read(65536)
            if not block:
                return digest.hexdigest()
            digest.update(block)
    raise BenchError(f"file grew past byte limit: {path}")


def read_json(path, limit=MAX_FILE_BYTES):
    require(path.stat().st_size <= limit, f"JSON exceeds byte limit: {path}")
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        require(len(data) <= limit, f"JSON grew beyond byte limit: {path}")
        value = json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_object)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise BenchError(f"invalid JSON in {path}: {exc}") from exc
    return value


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def write_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data)


def inventory(root):
    """Bound directory traversal, including empty directories, before copying."""
    require(root.is_dir() and not root.is_symlink(), f"missing directory: {root}")
    pending = [root]
    files = {}
    total = 0
    entries = 0
    for _ in range(MAX_FILES + 1):
        if not pending:
            return files
        folder = pending.pop()
        with os.scandir(folder) as children:
            for child in children:
                entries += 1
                require(entries <= MAX_FILES, f"too many entries in {root}")
                path = Path(child.path)
                relative = path.relative_to(root)
                require(len(relative.parts) <= 16, f"directory too deep: {path}")
                require(not child.is_symlink(), f"symlink is not an immutable input: {path}")
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    digest = sha_file(path)
                    size = path.stat().st_size
                    total += size
                    require(total <= MAX_TOTAL_BYTES, f"input exceeds byte limit: {root}")
                    files[str(relative)] = {"sha256": digest, "bytes": size}
    raise BenchError(f"directory count exceeds limit: {root}")


def copy_inventory(source, destination, expected, writable=False):
    destination.mkdir()
    for relative in sorted(expected):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        target.chmod(0o644 if writable else 0o444)
    require(inventory(destination) == expected, f"copied input changed: {source}")


@contextlib.contextmanager
def benchmark_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BenchError(f"benchmark lock is held: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def limit_child_artifacts(limit):
    # Limit each log/result file as well as bounding process lifetime. Exceeding
    # this limit is an abnormal child exit, never a silently truncated success.
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))


def run_process(command, cwd, evidence, timeout, env=None, artifact_limit=MAX_FILE_BYTES):
    """Kill/reap only our direct child; raw streams survive every exit path."""
    evidence.mkdir()
    started = time.monotonic()
    record = {"command": [str(part) for part in command], "cwd": str(cwd),
              "started_utc": utc_now(), "returncode": None, "status": "failed",
              "artifact_file_byte_limit": artifact_limit}
    with (evidence / "stdout.log").open("xb") as stdout:
        with (evidence / "stderr.log").open("xb") as stderr:
            try:
                with subprocess.Popen(command, cwd=cwd, env=env,
                                      stdin=subprocess.DEVNULL,
                                      stdout=stdout, stderr=stderr,
                                      preexec_fn=functools.partial(limit_child_artifacts, artifact_limit)) as process:
                    try:
                        record["returncode"] = process.wait(timeout=timeout)
                        record["status"] = "completed" if process.returncode == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        process.kill()
                        record["returncode"] = process.wait(timeout=5)
                        record["status"] = "timed_out"
                    except BaseException:
                        process.kill()
                        record["returncode"] = process.wait(timeout=5)
                        record["status"] = "cancelled"
                        record["process_wall_seconds"] = time.monotonic() - started
                        write_json(evidence / "process.json", record)
                        raise
            except OSError as exc:
                record["error"] = str(exc)
    record["process_wall_seconds"] = time.monotonic() - started
    write_json(evidence / "process.json", record)
    return record


def capture_command(command, directory, label, timeout=5):
    record = run_process(command, REPO, directory / label, timeout)
    for stream in ("stdout", "stderr"):
        path = directory / label / f"{stream}.log"
        with path.open("rb") as handle:
            record[stream] = handle.read(16384).decode("utf-8", errors="replace")
        record[f"{stream}_truncated"] = path.stat().st_size > 16384
    return record


def machine_provenance(output):
    commands = {"git_head": ["git", "rev-parse", "HEAD"],
                "git_status": ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                "source_patch": ["git", "diff", "--binary", "HEAD", "--", ".", ":(exclude).cache"],
                "rustc": ["rustc", "-Vv"], "cargo": ["cargo", "-V"],
                "rustup": ["rustup", "show", "active-toolchain"],
                "untracked_files": ["git", "ls-files", "--others", "--exclude-standard", "-z"]}
    records = {}
    for name, command in commands.items():
        records[name] = capture_command(command, output, name)
    lock = REPO / "Cargo.lock"
    return {"platform": platform.platform(), "uname": list(platform.uname()),
            "python": sys.version, "cpu_count": os.cpu_count(),
            "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
            "cargo_lock_sha256": sha_file(lock) if lock.is_file() else None,
            "supervisor_sha256": sha_file(Path(__file__).resolve()),
            "commands": records, "source_patch_sha256": sha_file(output / "source_patch/stdout.log"),
            "untracked_source_files": archive_untracked(output, records["untracked_files"])}


def archive_untracked(output, listing):
    if listing["status"] != "completed":
        return {"status": "unavailable", "reason": "git ls-files failed; raw error retained"}
    path = output / "untracked_files/stdout.log"
    require(path.stat().st_size <= 1024 * 1024, "untracked path listing exceeds limit")
    names = path.read_bytes().split(b"\0")
    require(len(names) <= MAX_FILES, "too many untracked source files to archive")
    archived = {}
    size = 0
    for raw in names:
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        require(not relative.is_absolute() and ".." not in relative.parts,
                "git returned an invalid source path")
        source = REPO / relative
        if source.resolve().is_relative_to(output.resolve()):
            continue  # Evidence is not an input source when --output is in the repo.
        digest = sha_file(source)
        size += source.stat().st_size
        require(size <= MAX_TOTAL_BYTES, "untracked source archive exceeds byte limit")
        target = output / "untracked-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o444)
        require(sha_file(target) == digest, "untracked source changed while archiving")
        archived[str(relative)] = digest
    return {"status": "archived", "sha256_by_path": archived}


def prepare_binary(args, output):
    artifact = None
    if args.binary is None:
        build = run_process(["cargo", "build", "--release", "--locked", "--example",
                             "evolution-bench", "--message-format=json-render-diagnostics"],
                            REPO, output / "build", args.timeout,
                            artifact_limit=MAX_BINARY_BYTES)
        require(build["status"] == "completed", "release build failed; see build logs")
        artifact = cargo_artifact(output / "build/stdout.log")
        source = Path(artifact["executable"])
    else:
        source = args.binary
    require(source.is_file() and os.access(source, os.X_OK), f"binary is not executable: {source}")
    digest = sha_file(source, MAX_BINARY_BYTES)
    target = output / "evolution-bench"
    shutil.copyfile(source, target)
    target.chmod(0o555)
    require(sha_file(target, MAX_BINARY_BYTES) == digest, "binary changed while being archived")
    return target, {"source": str(source), "archived": str(target), "sha256": digest,
                    "build": "supplied_binary_unverified_build" if args.binary else "release_locked",
                    "cargo_artifact": artifact}


def cargo_artifact(path):
    """Use Cargo's actual output, including target/configuration redirects."""
    require(path.stat().st_size <= MAX_FILE_BYTES, "Cargo artifact log exceeds limit")
    matches = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            require(index < MAX_FILES * 32, "too many Cargo artifact messages")
            message = object_value(json.loads(line), "Cargo artifact message")
            if message.get("reason") != "compiler-artifact":
                continue
            target = object_value(message.get("target"), "Cargo target")
            if target.get("name") == "evolution-bench" and "example" in target.get("kind", []):
                executable = message.get("executable")
                require(isinstance(executable, str) and Path(executable).is_absolute(),
                        "Cargo example artifact has no absolute executable path")
                matches.append(message)
    require(len(matches) == 1, "Cargo did not report exactly one evolution-bench executable")
    return matches[0]


def recv_exact(connection, count, deadline):
    require(0 <= count <= MAX_WS_BYTES, "backend WebSocket frame exceeds limit")
    chunks = []
    remaining = count
    for _ in range(MAX_WS_BYTES + 1):
        if remaining == 0:
            return b"".join(chunks)
        connection.settimeout(max(0.001, deadline - time.monotonic()))
        require(time.monotonic() < deadline, "backend observation deadline")
        block = connection.recv(min(remaining, 65536))
        require(bool(block), "backend closed connection")
        chunks.append(block)
        remaining -= len(block)
    raise BenchError("backend read exceeded iteration bound")


def backend_state():
    """Read the server's unsolicited init frame. Send no application commands."""
    try:
        deadline = time.monotonic() + 1.0
        with socket.create_connection(("127.0.0.1", 9001), timeout=0.2) as connection:
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            request = ("GET / HTTP/1.1\r\nHost: 127.0.0.1:9001\r\nUpgrade: websocket\r\n"
                       "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
                       f"Sec-WebSocket-Key: {key}\r\n\r\n")
            connection.sendall(request.encode("ascii"))
            header = bytearray()
            for _ in range(8192):
                header.extend(recv_exact(connection, 1, deadline))
                if header.endswith(b"\r\n\r\n"):
                    break
            require(header.endswith(b"\r\n\r\n"), "oversized backend handshake")
            accept = base64.b64encode(hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest())
            require(b" 101 " in header.split(b"\r\n")[0] and accept in header,
                    "invalid backend WebSocket handshake")
            first, second = recv_exact(connection, 2, deadline)
            require(first == 0x81 and not second & 0x80, "unsupported backend init frame")
            length = second & 0x7f
            if length == 126:
                length = struct.unpack("!H", recv_exact(connection, 2, deadline))[0]
            elif length == 127:
                length = struct.unpack("!Q", recv_exact(connection, 8, deadline))[0]
            init = json.loads(recv_exact(connection, length, deadline), parse_constant=reject_constant)
            require(init.get("type") == "init" and type(init.get("paused")) is bool,
                    "backend init has no paused state")
            return {"status": "observed", "evolving": not init["paused"],
                    "paused": init["paused"], "total_evals": init.get("totalEvals"),
                    "evals_per_second": init.get("evalsPerSec"),
                    "active_project": init.get("activeProject"),
                    "policy": "passive init only; no application messages"}
    except (OSError, BenchError, ValueError, AttributeError) as exc:
        return {"status": "unavailable", "evolving": None, "reason": str(exc)}


def telemetry(output, label):
    path = output / label
    path.mkdir()
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    nvidia = shutil.which("nvidia-smi")
    gpu = {"status": "unavailable", "reason": "nvidia-smi not installed"}
    if nvidia:
        query = "name,uuid,driver_version,utilization.gpu,clocks.current.graphics,temperature.gpu,power.draw"
        gpu = capture_command([nvidia, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                              path, "nvidia-smi", timeout=2)
        gpu["status"] = "available" if gpu["status"] == "completed" else "unavailable"
        gpu["fields"] = query.split(",")
    value = {"utc": utc_now(), "cpu_load_average": load, "cpu_count": os.cpu_count(),
             "gpu": gpu, "backend": backend_state(),
             "limitations": "point samples cannot prove absence of competing GPU activity"}
    write_json(path / "telemetry.json", value)
    return value


def fixture_file(root, value):
    require(isinstance(value, str) and value, "fixture path must be a nonempty string")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"fixture path must be relative: {value}")
    target = root / path
    require(target.is_file() and not target.is_symlink(), f"missing frozen file: {target}")
    require(target.resolve().is_relative_to(root.resolve()), "fixture escapes corpus directory")
    return target


def load_manifest(path):
    manifest = object_value(read_json(path, 2 * 1024 * 1024), "manifest")
    require(manifest.get("protocol_version") == 1, "unsupported manifest protocol_version")
    require(type(manifest.get("protocol_version")) is int, "manifest protocol must be integer")
    require(manifest.get("start_mode") == "mature_drawing_restart", "unsupported start mode")
    require(manifest.get("minimum_similarity_percent") == 97, "similarity gate must be 97 percent")
    integer(manifest.get("resolution"), 384, 384, "resolution")
    params = object_value(manifest.get("params"), "manifest.params")
    expected_work(params, 1)
    cases = manifest.get("cases")
    require(isinstance(cases, list) and 3 <= len(cases) <= MAX_CASES, "manifest needs 3..24 cases")
    normalized = []
    ids = set()
    projects = set()
    for case in cases:
        object_value(case, "case")
        name = case.get("id")
        require(isinstance(name, str) and CASE_ID.fullmatch(name), "invalid case_id")
        require(name not in ids, f"duplicate case_id: {name}")
        require(name in PROJECTS, f"unscored project: {name}")
        ids.add(name)
        projects.add(name)
        width = integer(case.get("width"), 2, 1024, "width")
        height = integer(case.get("height"), 2, 1024, "height")
        require(max(width, height) == 384, "scored cases require 384px longest side")
        validate_complexity(case.get("complexity"), f"{name}.complexity")
        digest_value(case.get("source_image_sha256"), "source_image_sha256")
        finite(case.get("stored_fitness_percent"), 0, 100, "stored_fitness_percent")
        finite(case.get("scalar_fitness_percent"), 0, 100, "scalar_fitness_percent")
        normalized.append(dict(case, case_id=name, project=name, params=params))
        for file_key, hash_key in (("reference_file", "reference_sha256"),
                                   ("drawing_file", "drawing_sha256")):
            file_path = fixture_file(path.parent, case.get(file_key))
            expected = digest_value(case.get(hash_key), hash_key)
            require(sha_file(file_path) == expected, f"fixture integrity mismatch: {file_path}")
            if file_key == "reference_file":
                require(file_path.stat().st_size == width * height * 4, "reference byte length differs from dimensions")
    require(projects == set(PROJECTS), "manifest must include ff, mona, darwin")
    return dict(manifest, cases=normalized)


def prepare_fixtures(manifest, output):
    cases = load_manifest(manifest)
    expected = inventory(manifest.parent)
    snapshot = output / "fixtures"
    copy_inventory(manifest.parent, snapshot, expected)
    frozen = snapshot / manifest.name
    require(load_manifest(frozen) == cases, "fixture snapshot manifest changed")
    return frozen, cases, expected


def expected_work(params, batches):
    iterations = integer(params.get("gpuBatchIters"), 1, 4096, "params.gpuBatchIters") * batches
    chains = integer(params.get("chainCount"), 1, 4096, "params.chainCount")
    offspring = integer(params.get("lambda"), 1, 64, "params.lambda")
    return {"batches": batches, "iterations": iterations,
            "evaluations": iterations * chains * offspring}


def validate_complexity(value, name):
    value = object_value(value, name)
    triangles = integer(value.get("triangles"), 1, 1000, f"{name}.triangles")
    require(value.get("vertices") == triangles * 3, f"{name}.vertices differs from packed triangles")
    integer(value.get("tile_count"), 1, 16384, f"{name}.tile_count")
    require(type(value.get("overflowed_tiles")) is int and value["overflowed_tiles"] == 0,
            f"{name} has tile overflow or missing overflow count")
    for field in ("triangle_area", "bbox_area"):
        area = object_value(value.get(field), f"{name}.{field}")
        bounds = [finite(area.get(key), 0, 1, f"{name}.{field}.{key}")
                  for key in ("min", "median", "p95", "max")]
        require(bounds == sorted(bounds), f"{name}.{field} quantiles are not ordered")
    overlap = object_value(value.get("tile_overlaps"), f"{name}.tile_overlaps")
    bounds = [integer(overlap.get(key), 0, 256, f"{name}.tile_overlaps.{key}")
              for key in ("min", "p50", "p95", "max")]
    require(bounds == sorted(bounds), f"{name}.tile_overlaps quantiles are not ordered")


def validate_quality(value, case, initial, name):
    require(isinstance(value, list) and len(value) == case["params"]["chainCount"],
            f"{name} needs one numerical check per active parent")
    pixels = case["width"] * case["height"]
    for index, quality in enumerate(value):
        object_value(quality, f"{name}[{index}]")
        require(type(quality.get("chain")) is int and quality["chain"] == index,
                f"{name} chain indices differ")
        integer(quality.get("triangles"), 1, 1000, f"{name}.triangles")
        errors = [integer(quality.get(key), 0, pixels * 765, f"{name}.{key}")
                  for key in ("stored_error", "framebuffer_error", "scalar_error")]
        integer(quality.get("differing_pixels"), 0, pixels, f"{name}.differing_pixels")
        audit = object_value(quality.get("pixel_audit"), f"{name}.pixel_audit")
        integer(audit.get("outside_nominal_quantum"), 0, pixels, f"{name}.outside_nominal_quantum")
        integer(audit.get("widest_channel_envelope"), 0, 255, f"{name}.widest_channel_envelope")
        require(type(audit.get("unexplained_pixels")) is int and audit["unexplained_pixels"] == 0,
                f"{name} pixel audit has unexplained pixels or no unexplained-pixel count")
        maximum = integer(quality.get("max_channel_delta"), 0, 255, f"{name}.max_channel_delta")
        delta = integer(quality.get("total_channel_delta"), 0, pixels * 3 * 255, f"{name}.total_channel_delta")
        require(errors[0] == errors[1] and abs(errors[0] - errors[2]) <= delta,
                f"{name} cached, framebuffer and independent error disagree")
        require(delta <= quality["differing_pixels"] * 3 * maximum,
                f"{name} pixel delta metrics disagree")
        fitness = finite(quality.get("stored_fitness"), 0, 100, f"{name}.stored_fitness")
        expected = finite(quality.get("expected_stored_fitness"), 0, 100, f"{name}.expected_stored_fitness")
        finite(quality.get("scalar_fitness"), 0, 100, f"{name}.scalar_fitness")
        require(abs(fitness - expected) <= 0.00002, f"{name} stored fitness disagrees with error")
        require(not initial or fitness > 97, f"{name} initial parent must exceed 97 percent")


def validate_quality_record(record, case):
    validate_complexity(record.get("initial_complexity"), "initial_complexity")
    require(record["initial_complexity"] == case["complexity"], "initial complexity differs from frozen case")
    initial, final = record.get("initial_quality"), record.get("final_quality")
    validate_quality(initial, case, True, "initial_quality")
    validate_quality(final, case, False, "final_quality")
    require(all(quality["triangles"] == record["initial_complexity"]["triangles"]
                for quality in initial), "initial quality/complexity mismatch")
    complexity = record.get("final_complexity")
    require(isinstance(complexity, list) and len(complexity) == len(final), "missing final complexity per parent")
    for index, quality in enumerate(final):
        validate_complexity(complexity[index], f"final_complexity[{index}]")
        require(quality["triangles"] == complexity[index]["triangles"], "final quality/complexity mismatch")
        require(record["final_fitness_percent"][index] == quality["stored_fitness"],
                "quality/hash mismatch: final fitness differs from checked quality")
        require(quality["stored_fitness"] + 0.00002 >= initial[index]["stored_fitness"],
                "final parent fitness regressed against its initial parent")
    require(record.get("start_mode") == "mature_drawing_restart", "missing or wrong restart mode")
    samples = integer(record.get("timestamp_samples"), 1, 2**32, "timestamp_samples")
    stages = record.get("sampled_gpu_stage_totals_ns")
    require(isinstance(stages, list) and len(stages) == 4, "missing sampled GPU stage timings")
    for stage in stages:
        finite(stage, 0, samples * 1e15, "sampled_gpu_stage_totals_ns")


def validate_result(record, case, args, diagnostic, manifest_hash):
    """Validate an ordinary scored timing or unscored diagnostic TrialRecord."""
    purpose = "diagnostic" if diagnostic else "timing"
    return _validate_trial_record(record, case, args, diagnostic, manifest_hash, purpose)


def validate_qualification(record, case, args, manifest_hash):
    """Validate an unscored qualification, even when diagnostic is false."""
    return _validate_trial_record(record, case, args, False, manifest_hash, "qualification")


def _validate_trial_record(record, case, args, diagnostic, manifest_hash, purpose):
    object_value(record, "result")
    require(record.get("timestamp_mode", "stages") == "stages"
            and record.get("batch_compute_envelope") is None,
            "profiling records cannot enter the reference timing score")
    require(type(record.get("protocol_version")) is int and record["protocol_version"] == 1,
            "unsupported result protocol")
    require(record.get("status") == "completed", "trial did not complete validly")
    require(type(record.get("correctness")) is bool and record["correctness"], "numerical correctness failed")
    require(record.get("errors") == [] and type(record.get("errors")) is list, "trial reported errors")
    require(type(record.get("error_count")) is int and record["error_count"] == 0, "trial error_count is not zero")
    for key, expected in (("case_id", case["case_id"]), ("seed", args.seed),
                          ("batches", args.batches), ("diagnostic", diagnostic)):
        require(type(record.get(key)) is type(expected) and record[key] == expected,
                f"result {key} differs from requested trial")
    require(record.get("fixture_manifest_sha256") == manifest_hash, "fixture manifest hash mismatch")
    for key in ("initial_state_sha256", "final_state_sha256"):
        digest_value(record.get(key), key)
    require(record["initial_state_sha256"] != record["final_state_sha256"],
            "no evolution state changed despite reported completed work")
    params = object_value(record.get("params"), "result.params")
    if "params" in case:
        require(params == case["params"], "effective parameters differ from frozen case")
    counters = object_value(record.get("counters"), "result.counters")
    for key, expected in expected_work(params, args.batches).items():
        for phase in ("submitted", "completed"):
            field = f"{phase}_{key}"
            require(type(counters.get(field)) is int and counters[field] == expected,
                    f"wrong completed work: {field}, expected {expected}, got {counters.get(field)}")
    elapsed = finite(record.get("elapsed_seconds"), 1e-9, args.timeout, "elapsed_seconds")
    finite(record.get("drain_seconds"), 0, elapsed, "drain_seconds")
    if purpose != "timing":
        require("evals_per_second" in record and record["evals_per_second"] is None,
                f"{purpose} evals_per_second must be null")
    else:
        rate = finite(record.get("evals_per_second"), 1e-9, 1e18, "evals_per_second")
        require(math.isclose(rate, counters["completed_evaluations"] / elapsed, rel_tol=1e-6),
                "throughput does not match completed work / elapsed")
    finite(record.get("initial_similarity_percent"), 97.000000001, 100, "initial_similarity_percent")
    fitness = record.get("final_fitness_percent")
    require(isinstance(fitness, list) and len(fitness) == params["chainCount"], "missing final per-chain fitness")
    for value in fitness:
        finite(value, 0, 100, "final_fitness_percent")
    validate_quality_record(record, case)
    object_value(record.get("gpu"), "gpu provenance")
    if diagnostic:
        integer(record.get("validation_max_tile_occupancy"), 0, 256, "validation_max_tile_occupancy")
    return record


def replay_errors(a, b):
    """A repeated timing seed is one search trajectory, never an independent seed."""
    fields = ("initial_state_sha256", "final_state_sha256", "initial_similarity_percent",
              "final_fitness_percent", "initial_complexity", "final_complexity",
              "initial_quality", "final_quality", "params", "gpu", "counters")
    return [f"replay/quality mismatch: {key}" for key in fields if a.get(key) != b.get(key)]


def verify_saved_state(output, record):
    state = object_value(read_json(output / "state.json"), "state.json")
    fields = ("parents", "offspring", "parent_total_errors", "parent_framebuffers",
              "offspring_errors", "chain_update_flags", "fitness_packed")
    for field in fields:
        require(isinstance(state.get(field), list), f"missing canonical state field: {field}")
    canonical = [state[field] for field in fields]
    encoded = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    require(hashlib.sha256(encoded).hexdigest() == record["final_state_sha256"],
            "quality/hash mismatch: final_state_sha256 differs from saved canonical state")
    require(state.get("counters") == record["counters"], "saved state counters differ from result")
    require(len(state["parents"]) == record["params"]["chainCount"], "saved parent count differs")
    require(state["parent_total_errors"] == [quality["stored_error"] for quality in record["final_quality"]],
            "quality/hash mismatch: saved errors differ from checked quality")
    for index, parent in enumerate(state["parents"]):
        object_value(parent, "saved parent")
        require(parent.get("polygon_count") == record["final_quality"][index]["triangles"],
                "quality/hash mismatch: saved packed triangles differ")
        bits = integer(parent.get("fitness_bits"), 0, 2**32 - 1, "saved fitness_bits")
        fitness = struct.unpack("<f", struct.pack("<I", bits))[0]
        require(math.isclose(fitness, record["final_fitness_percent"][index], abs_tol=0.00002),
                "quality/hash mismatch: saved packed fitness differs")
    drawings = read_json(output / "drawings.json")
    require(isinstance(drawings, list) and len(drawings) == len(state["parents"]),
            "missing final drawings per parent")


def execute_trial(args, binary, manifest, case, trial_dir, diagnostic, cache, fixture_hashes):
    trial_dir.mkdir()
    cwd = trial_dir / "cwd"
    copy_inventory(cache, cwd, inventory(cache), writable=True)
    output = trial_dir / "output"
    before = telemetry(trial_dir, "before")
    entry = {"case_id": case["case_id"], "project": case["project"],
             "diagnostic": diagnostic, "errors": [], "before": before,
             "directory": str(trial_dir), "valid": False}
    try:
        require(inventory(manifest.parent) == fixture_hashes, "fixture integrity changed before trial")
        command = [str(binary), "trial", "--manifest", str(manifest), "--case", case["case_id"],
                   "--seed", str(args.seed), "--batches", str(args.batches),
                   "--warmup", str(args.warmup), "--diagnostic", str(diagnostic).lower(),
                   "--output", str(output)]
        process = run_process(command, cwd, trial_dir / "process", args.timeout)
        entry["process"] = process
        entry["status"] = process["status"]
        if (output / "result.json").is_file():
            entry["result"] = read_json(output / "result.json", 2 * 1024 * 1024)
        require(process["status"] == "completed", f"subprocess {process['status']}, exit {process['returncode']}")
        require("result" in entry, "missing result.json")
        validate_result(entry["result"], case, args, diagnostic, sha_file(manifest))
        require((output / "state.json").is_file(), "missing state.json")
        verify_saved_state(output, entry["result"])
        entry["artifacts"] = inventory(output)
        require(inventory(manifest.parent) == fixture_hashes, "fixture integrity changed during trial")
        entry["valid"] = True
    except (BenchError, OSError) as exc:
        entry["errors"].append(str(exc))
        entry["status"] = "invalid" if entry.get("status") == "completed" else entry.get("status", "invalid")
    except KeyboardInterrupt:
        entry["errors"].append("subprocess cancelled")
        entry["status"] = "cancelled"
    entry["after"] = telemetry(trial_dir, "after")
    write_json(trial_dir / "supervisor.json", entry)
    return entry


def distribution(values):
    if not values:
        return None
    mean = statistics.mean(values)
    return {"n": len(values), "median": statistics.median(values), "min": min(values),
            "max": max(values), "range": max(values) - min(values), "mean": mean,
            "cv": statistics.pstdev(values) / mean if mean else None, "raw": values}


def case_statistics(case, entries, comparisons):
    normal = [entry for entry in entries if not entry["diagnostic"]]
    accepted = [entry["result"] for entry in normal if entry["valid"]]
    ratios = []
    for comparison in comparisons:
        if comparison["valid"]:
            a, b = comparison["a_rate"], comparison["b_rate"]
            ratios.append(a / b)
    return {"case_id": case["case_id"], "project": case["project"],
            "valid": len(accepted) == len(normal) and all(pair["valid"] for pair in comparisons),
            "error_count": sum(len(pair["errors"]) for pair in comparisons),
            "elapsed_seconds": distribution([value["elapsed_seconds"] for value in accepted]),
            "evals_per_second": distribution([value["evals_per_second"] for value in accepted]),
            "aa_ratios": distribution(ratios),
            "completed_evaluations": sum(value["counters"]["completed_evaluations"] for value in accepted),
            "comparisons": comparisons}


def project_statistics(cases):
    projects = []
    for name in PROJECTS:
        selected = [case for case in cases if case["project"] == name]
        valid = bool(selected) and all(case["valid"] for case in selected)
        rates = [case["evals_per_second"]["median"] for case in selected
                 if case["valid"] and case["evals_per_second"]]
        samples = weighted_samples(selected, "evals_per_second") if valid else []
        ratios = weighted_samples(selected, "aa_ratios") if valid else []
        projects.append({"project": name, "weight": 1 / 3, "valid": valid,
                         "error_count": sum(case["error_count"] for case in selected),
                         "case_ids": [case["case_id"] for case in selected],
                         "case_weight_within_project": 1 / len(selected) if selected else None,
                         "evals_per_second": distribution(samples), "aa_ratios": distribution(ratios),
                         "geomean_case_median_evals_per_second": geometric_mean(rates) if valid else None})
    valid = all(project["valid"] for project in projects)
    aggregate = geometric_mean([project["geomean_case_median_evals_per_second"]
                                for project in projects]) if valid else None
    return {"weighting": "equal projects, equal cases within each project, geometric means",
            "projects": projects, "aggregate_evals_per_second": aggregate}


def weighted_samples(cases, field):
    assert cases and all(case["valid"] for case in cases)
    samples = [case[field]["raw"] for case in cases]
    assert all(len(value) == len(samples[0]) for value in samples)
    return [geometric_mean([value[index] for value in samples]) for index in range(len(samples[0]))]


def geometric_mean(values):
    return math.exp(statistics.mean([math.log(value) for value in values])) if values else None


def compare_group(case, index, entries):
    assert len(entries) == 3
    a, b, diagnostic = entries
    errors = []
    for entry in entries:
        errors.extend(entry["errors"])
    if all(entry["valid"] for entry in entries):
        errors.extend(replay_errors(a["result"], b["result"]))
        errors.extend(replay_errors(a["result"], diagnostic["result"]))
    pair = {"case_id": case["case_id"], "pair": index, "valid": not errors,
            "errors": errors, "order": [entry["label"] for entry in entries],
            "directories": [entry["directory"] for entry in entries]}
    if not errors:
        pair["a_rate"] = a["result"]["evals_per_second"]
        pair["b_rate"] = b["result"]["evals_per_second"]
    else:
        for entry in entries:
            entry["valid"] = False
    return pair


def run_suite(args, binary, manifest, content, hashes, output, summary):
    seed_dir = output / "cache-seed"
    seed_hashes = inventory(args.cache_seed) if args.cache_seed else {}
    if args.cache_seed:
        copy_inventory(args.cache_seed, seed_dir, seed_hashes)
    else:
        seed_dir.mkdir()
    summary["cache_policy"] = {"kind": "identical copied seed per process", "files": seed_hashes}
    trials_dir = output / "trials"
    trials_dir.mkdir()
    pairs = args.pairs if args.command == "baseline" else 1
    trial_count = len(content["cases"]) * pairs * 3
    require(trial_count <= MAX_TRIALS, "trial count exceeds limit")
    all_entries, all_cases = [], []
    summary["trials"] = all_entries
    summary["cases"] = all_cases
    for case in content["cases"]:
        case_entries, comparisons = [], []
        for index in range(pairs):
            group = []
            for label, diagnostic in (("A", False), ("B", False), ("diagnostic", True)):
                directory = trials_dir / f"{case['case_id']}-{index:02d}-{label}"
                print(f"[{len(all_entries) + 1}/{trial_count}] {case['case_id']} pair {index + 1} {label}",
                      file=sys.stderr, flush=True)
                entry = execute_trial(args, binary, manifest, case, directory, diagnostic, seed_dir, hashes)
                print(f"  {entry['status']}; {len(entry['errors'])} errors", file=sys.stderr, flush=True)
                entry["label"] = label
                group.append(entry)
                all_entries.append(entry)
                case_entries.append(entry)
                if entry["status"] == "cancelled":
                    raise KeyboardInterrupt
            comparison = compare_group(case, index, group)
            comparisons.append(comparison)
            summary["errors"].extend(comparison["errors"])
        all_cases.append(case_statistics(case, case_entries, comparisons))
    summary["trials"] = all_entries
    summary["cases"] = all_cases
    summary["aggregation"] = project_statistics(all_cases)
    summary["error_count"] = len(summary["errors"])
    summary["completed_work"] = completed_work(all_entries)
    summary["valid"] = not summary["errors"]
    observations = [entry[phase]["backend"] for entry in all_entries for phase in ("before", "after")]
    evolving = sum(observation.get("evolving") is True for observation in observations)
    summary["contamination"] = {"backend_evolving_observations": evolving,
                                "backend_unavailable_observations": sum(
                                    observation.get("status") != "observed" for observation in observations),
                                "continuous_gpu_activity_observation": "unavailable"}
    if args.command == "baseline" and evolving:
        summary["measurement_context"] = "baseline_contended"


def completed_work(entries):
    totals = {key: 0 for key in ("completed_batches", "completed_iterations", "completed_evaluations")}
    for entry in entries:
        record = entry.get("result")
        counters = record.get("counters", {}) if isinstance(record, dict) else {}
        if not isinstance(counters, dict):
            continue
        for key in totals:
            value = counters.get(key)
            if type(value) is int and value >= 0:
                totals[key] += value
    return {"reported_including_invalid_and_diagnostic": totals,
            "policy": "invalid/diagnostic work is retained for accounting and excluded from timing score"}


def freeze(args, binary, output, summary):
    require(not args.fixtures.exists(), f"fixture destination already exists: {args.fixtures}")
    require(not args.fixtures.is_symlink(), "fixture destination is a symlink")
    cwd = output / "cwd"
    cwd.mkdir()
    process = run_process([str(binary), "freeze", "--projects-dir", str(args.projects_dir),
                           "--fixtures", str(args.fixtures), "--resolution", str(args.resolution)],
                          cwd, output / "freeze-process", args.timeout)
    summary["freeze_process"] = process
    require(process["status"] == "completed", "freeze failed; see freeze-process logs")
    manifest = args.fixtures / "manifest.json"
    content = load_manifest(manifest)
    hashes = inventory(args.fixtures)
    summary.update({"manifest": str(manifest), "manifest_sha256": sha_file(manifest),
                    "fixture_inventory": hashes, "cases": content["cases"], "valid": True,
                    "qualification": "import only; GPU qualification requires validate"})


def new_summary(args):
    return {"protocol_version": 1, "command": args.command, "started_utc": utc_now(),
            "valid": False, "errors": [], "error_count": 0,
            "clean_baseline": None if args.command == "baseline" else False,
            "isolation_state": "unverified",
            "measurement_context": "development_contended" if args.command == "validate"
            else "reference_capture_unverified_isolation",
            "verdict": "no throughput verdict", "limiter": "unmeasured",
            "campaign_frozen": False, "optimization_claim": False,
            "timing_policy": {"normal": "engine completed-work wall time, including final drain",
                              "excluded": "build, setup, warmup, post-drain verification, diagnostic replay",
                              "gpu_stages": "sampled attribution only, not complete GPU wall time"},
            "seed_schedule": {"seed": getattr(args, "seed", None),
                              "repetitions_are_independent_search_seeds": False},
            "isolation_policy": "preserve competing work; record contamination; point telemetry cannot establish idle",
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}}


def validate_paths(args):
    output = args.output.resolve()
    lock = args.lock.resolve()
    require(not lock.is_relative_to(output), "lock path must be outside the new output directory")
    inputs = [args.fixtures, args.projects_dir] if args.command == "freeze" else [args.manifest.parent]
    if getattr(args, "cache_seed", None):
        inputs.append(args.cache_seed)
    for source in inputs:
        root = source.resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output),
                f"output and input directory must be disjoint: {source}")
        require(not lock.is_relative_to(root), f"lock path must be outside input directory: {source}")


def execute(args):
    validate_paths(args)
    summary = new_summary(args)
    with benchmark_lock(args.lock):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.mkdir()  # Exclusive creation; never reuse or clear an old bundle.
        try:
            if args.command == "freeze":
                require(not args.fixtures.exists() and not args.fixtures.is_symlink(),
                        f"fixture destination already exists: {args.fixtures}")
            summary["provenance"] = machine_provenance(args.output)
            binary, identity = prepare_binary(args, args.output)
            summary["binary"] = identity
            if args.command == "freeze":
                freeze(args, binary, args.output, summary)
            else:
                manifest, content, hashes = prepare_fixtures(args.manifest, args.output)
                summary.update({"manifest": str(manifest), "manifest_sha256": sha_file(manifest),
                                "fixture_inventory": hashes})
                run_suite(args, binary, manifest, content, hashes, args.output, summary)
            require(sha_file(binary, MAX_BINARY_BYTES) == identity["sha256"], "archived executable changed during run")
        except (BenchError, OSError, ValueError) as exc:
            summary["errors"].append(str(exc))
            summary["valid"] = False
        except KeyboardInterrupt:
            summary["errors"].append("supervisor cancelled")
            summary["valid"] = False
            summary["status"] = "cancelled"
        finally:
            if "trials" in summary:
                summary["completed_work"] = completed_work(summary["trials"])
            summary["error_count"] = len(summary["errors"])
            summary["finished_utc"] = utc_now()
            summary["status"] = summary.get("status", "completed" if summary["valid"] else "invalid")
            write_json(args.output / "summary.json", summary)
    print(json.dumps({"summary": str(args.output / "summary.json"), "valid": summary["valid"],
                      "error_count": summary["error_count"], "completed_work": summary.get("completed_work"),
                      "clean_baseline": summary["clean_baseline"], "verdict": "no throughput verdict"}, allow_nan=False))
    return 0 if summary["valid"] else 1


def bounded_int(low, high):
    def parse(value):
        try:
            return integer(int(value), low, high, "argument")
        except (ValueError, BenchError) as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return parse


def timeout_arg(value):
    try:
        return finite(float(value), 0.05, 3600, "timeout")
    except (ValueError, BenchError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = cli.add_subparsers(dest="command", required=True)
    for command in ("freeze", "validate", "baseline"):
        sub = modes.add_parser(command, help={"freeze": "import ff/mona/darwin without a GPU run",
                    "validate": "short A/B and diagnostic replay; development timings only",
                    "baseline": "capture a reference A/A series; run later on a quiet machine"}[command])
        sub.add_argument("--binary", type=Path, help="existing executable; otherwise build release example once")
        sub.add_argument("--output", type=Path, help="new evidence directory; existing paths are refused")
        sub.add_argument("--lock", type=Path, default=DEFAULT_ROOT / "benchmark.lock", help="shared exclusive POSIX lock")
        sub.add_argument("--timeout", type=timeout_arg, default=600.0, help="per-process seconds, 0.05..3600 (default 600)")
        if command == "freeze":
            sub.add_argument("--projects-dir", type=Path, default=REPO / "projects")
            sub.add_argument("--fixtures", type=Path, default=DEFAULT_ROOT / "fixtures/v1")
            sub.add_argument("--resolution", type=bounded_int(384, 384), default=384,
                             help="longest side, fixed to 384 for this slice")
        else:
            sub.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "fixtures/v1/manifest.json")
            sub.add_argument("--seed", type=bounded_int(0, 2**64 - 1), default=7)
            sub.add_argument("--batches", type=bounded_int(1, 4096), default=64 if command == "baseline" else 2)
            sub.add_argument("--warmup", type=bounded_int(0, 16), default=2)
            sub.add_argument("--cache-seed", type=Path, help="immutable cache directory copied identically for each process")
            if command == "baseline":
                sub.add_argument("--pairs", type=bounded_int(1, 20), default=5)
    return cli


def parse_args(argv):
    args = parser().parse_args(argv)
    if args.output is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        args.output = DEFAULT_ROOT / f"{args.command}-{stamp}-{os.getpid()}"
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, Path(os.path.abspath(value.expanduser())))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return execute(args)
    except (BenchError, OSError) as exc:
        print(f"evolution-bench: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
