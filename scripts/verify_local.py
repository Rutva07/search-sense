"""Start real Redis and HTTP API, run all tests, benchmark, and clean up processes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import httpx
import redis

ROOT = Path(__file__).resolve().parents[1]


def wait_ready(probe, process, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Service exited during startup; see runtime logs")
        try:
            if probe():
                return
        except (redis.RedisError, httpx.HTTPError):
            pass
        time.sleep(.1)
    raise TimeoutError("Service startup timed out")


def main():
    executable = os.environ.get("SEARCHSENSE_REDIS_EXECUTABLE") or shutil.which("redis-server")
    if not executable:
        raise RuntimeError("Install redis-server or set SEARCHSENSE_REDIS_EXECUTABLE")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    namespace = "verify-" + uuid.uuid4().hex
    env = os.environ.copy()
    env.update(REDIS_URL="redis://127.0.0.1:16379/0", SEARCHSENSE_TEST_REDIS_URL="redis://127.0.0.1:16379/15",
               SEARCHSENSE_ARTIFACTS=str(ROOT / "artifacts"), SEARCHSENSE_NAMESPACE=namespace)
    processes = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temp, (reports / "redis_runtime.log").open("w") as redis_log, (reports / "api_runtime.log").open("w") as api_log:
        try:
            proc = subprocess.Popen([executable, "--port", "16379", "--save", "", "--appendonly", "no", "--dir", temp], env=env, stdout=redis_log, stderr=subprocess.STDOUT)
            processes.append(proc)
            client = redis.Redis.from_url(env["REDIS_URL"])
            wait_ready(client.ping, proc)
            version = client.info("server")["redis_version"]
            with (reports / "tests.txt").open("w") as log:
                subprocess.run([sys.executable, "-m", "pytest", "-q", "-ra"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
            # An unexpected skip is a failure in this verification workflow.
            test_output = (reports / "tests.txt").read_text()
            if "skipped" in test_output:
                raise RuntimeError("Verification requires all Redis/API tests to run")
            proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "searchsense.api:app", "--host", "127.0.0.1", "--port", "18000", "--workers", "1"], cwd=ROOT, env=env, stdout=api_log, stderr=subprocess.STDOUT)
            processes.append(proc)
            wait_ready(lambda: httpx.get("http://127.0.0.1:18000/health", timeout=1, trust_env=False).status_code == 200, proc)
            subprocess.run([sys.executable, str(ROOT / "scripts/benchmark_http.py"), "--url", "http://127.0.0.1:18000", "--redis-url", env["REDIS_URL"], "--namespace", namespace], cwd=ROOT, env=env, check=True, timeout=240)
            with (reports / "example_client.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT / "scripts/example_client.py"), "--url", "http://127.0.0.1:18000"], cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            report = {"status": "passed", "redis_version": version, "test_output": test_output,
                      "http_benchmark": "reports/http_latency.json", "seconds": time.monotonic()-started,
                      "docker_compose_executed": False}
            (reports / "verification.json").write_text(json.dumps(report, indent=2))
            print(test_output)
        finally:
            for proc in reversed(processes):
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
