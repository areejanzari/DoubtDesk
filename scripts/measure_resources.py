import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    "What is photosynthesis?",
    "Explain the structure of an atom.",
    "How does a lens form an image?",
    "What is an alkyl halide?",
    "Describe Newton's second law.",
]


class Sampler:
    def __init__(self, pid: int, interval: float = 0.25):
        self.process = psutil.Process(pid)
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._known = {}

    def _tree(self):
        try:
            current = [self.process, *self.process.children(recursive=True)]
        except psutil.NoSuchProcess:
            return []
        tree = []
        for proc in current:
            tree.append(self._known.setdefault(proc.pid, proc))
        return tree

    def rss_mb(self):
        total = 0
        for proc in self._tree():
            try:
                total += proc.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total / 1024 / 1024

    def threads(self):
        total = 0
        for proc in self._tree():
            try:
                total += proc.num_threads()
            except psutil.NoSuchProcess:
                pass
        return total

    def _run(self):
        for proc in self._tree():
            try:
                proc.cpu_percent(None)
            except psutil.NoSuchProcess:
                pass
        while not self._stop.wait(self.interval):
            cpu = 0.0
            for proc in self._tree():
                try:
                    cpu += proc.cpu_percent(None)
                except psutil.NoSuchProcess:
                    pass
            self.samples.append((cpu, self.rss_mb(), self.threads()))

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()

    def summary(self):
        if not self.samples:
            return {}
        cpus = [s[0] for s in self.samples]
        return {
            "cpu_avg_pct": round(statistics.mean(cpus), 1),
            "cpu_peak_pct": round(max(cpus), 1),
            "rss_peak_mb": round(max(s[1] for s in self.samples), 1),
            "threads_peak": max(s[2] for s in self.samples),
        }


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 3)


async def login(client: httpx.AsyncClient, index: int) -> str:
    response = await client.post("/auth/login", json={"identifier": f"bench_{index}", "password": "bench-password-123"})
    response.raise_for_status()
    return response.json()["csrf_token"]


async def wait_document(client, csrf, doc_id, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        document = (await client.get(f"/documents/{doc_id}/status")).json()
        if document["status"] == "failed":
            raise RuntimeError("Document ingestion failed")
        if document["status"] == "completed":
            return document
        await asyncio.sleep(0.5)
    raise TimeoutError("Document did not finish processing")


async def upload(base_url, pdf: Path, sampler: Sampler):
    async with httpx.AsyncClient(base_url=base_url, timeout=900) as client:
        csrf = await login(client, 0)
        started = time.perf_counter()
        with sampler, pdf.open("rb") as handle:
            response = await client.post("/documents/upload", files={"file": (pdf.name, handle, "application/pdf")},
                                         headers={"X-CSRF-Token": csrf})
            response.raise_for_status()
            body = response.json()
            response_seconds = time.perf_counter() - started
            if body["status"] == "processing":
                await wait_document(client, csrf, body["doc_id"])
        total = time.perf_counter() - started
        return body["doc_id"], {"response_s": round(response_seconds, 2), "total_s": round(total, 2), **sampler.summary()}


async def share_document(base_url, pdf: Path, users: int):
    async with httpx.AsyncClient(base_url=base_url, timeout=900) as client:
        for index in range(1, users):
            client.cookies.clear()
            csrf = await login(client, index)
            with pdf.open("rb") as handle:
                response = await client.post("/documents/upload", files={"file": (pdf.name, handle, "application/pdf")},
                                             headers={"X-CSRF-Token": csrf})
                response.raise_for_status()


async def ask_worker(base_url, index, doc_id, rounds, ttft, totals, errors, health_latencies):
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
        csrf = await login(client, index)
        conversation_id = None
        for round_index in range(rounds):
            payload = {"question": QUESTIONS[(index + round_index) % len(QUESTIONS)], "doc_ids": [doc_id]}
            if conversation_id:
                payload["conversation_id"] = conversation_id
            started = time.perf_counter()
            first = None
            async with client.stream("POST", "/ask/stream", json=payload, headers={"X-CSRF-Token": csrf}) as response:
                event = None
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: "):
                        data = json.loads(line[6:])
                        if event == "status" and data.get("conversation_id"):
                            conversation_id = data["conversation_id"]
                        if event == "token" and first is None:
                            first = time.perf_counter() - started
                        if event == "error":
                            errors.append(data)
                        if event == "complete":
                            conversation_id = data["conversation_id"]
            totals.append(time.perf_counter() - started)
            if first is not None:
                ttft.append(first)


async def health_probe(base_url, stop: asyncio.Event, latencies):
    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
        while not stop.is_set():
            started = time.perf_counter()
            await client.get("/health")
            latencies.append(time.perf_counter() - started)
            await asyncio.sleep(0.2)


async def load(base_url, doc_id, concurrency, rounds, sampler):
    ttft, totals, errors, health = [], [], [], []
    stop = asyncio.Event()
    started = time.perf_counter()
    with sampler:
        probe = asyncio.create_task(health_probe(base_url, stop, health))
        await asyncio.gather(*(ask_worker(base_url, i, doc_id, rounds, ttft, totals, errors, health)
                               for i in range(concurrency)))
        stop.set()
        await probe
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "requests": len(totals),
        "errors": len(errors),
        "wall_s": round(wall, 2),
        "throughput_rps": round(len(totals) / wall, 2),
        "ttft_p50_s": pct(ttft, 0.5),
        "ttft_p95_s": pct(ttft, 0.95),
        "total_p50_s": pct(totals, 0.5),
        "total_p95_s": pct(totals, 0.95),
        "health_p95_ms": round((pct(health, 0.95) or 0) * 1000, 1),
        **sampler.summary(),
    }


def start_server(port, users, workdir):
    env = {**os.environ, "BENCH_PORT": str(port), "BENCH_USERS": str(users), "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "bench_server.py")], cwd=workdir, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    threading.Thread(target=lambda: [lines.append(line) for line in process.stdout], daemon=True).start()
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Bench server exited:\n" + "".join(lines[-40:]))
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return process, time.perf_counter() - started, lines
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    process.kill()
    raise TimeoutError("Bench server did not become healthy:\n" + "".join(lines[-40:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(ROOT / "test_pdfs" / "keph203.pdf"))
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    users = max(args.concurrency)
    base_url = f"http://127.0.0.1:{args.port}"
    with tempfile.TemporaryDirectory(prefix="doubtdesk_bench_", ignore_cleanup_errors=True) as workdir:
        process, startup_s, lines = start_server(args.port, users, workdir)
        try:
            sampler = Sampler(process.pid)
            time.sleep(3)
            report = {"label": args.label, "cpu_count": psutil.cpu_count(),
                      "startup_s": round(startup_s, 1), "idle_rss_mb": round(sampler.rss_mb(), 1),
                      "idle_threads": sampler.threads()}
            doc_id, report["upload"] = asyncio.run(upload(base_url, Path(args.pdf), sampler))
            asyncio.run(share_document(base_url, Path(args.pdf), users))
            report["load"] = [asyncio.run(load(base_url, doc_id, c, args.rounds, sampler)) for c in args.concurrency]
            report["final_rss_mb"] = round(sampler.rss_mb(), 1)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
