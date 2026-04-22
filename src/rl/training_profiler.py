from __future__ import annotations

import cProfile
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Iterator


@dataclass
class ProfileSectionStats:
    total_seconds: float = 0.0
    calls: int = 0


class RunProfiler:
    def __init__(
        self,
        *,
        enabled: bool,
        top_n: int,
        output_path: Path | None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.top_n = int(top_n)
        self.output_path = output_path
        self._cpu_profiler = cProfile.Profile() if self.enabled else None
        self._sections: dict[str, ProfileSectionStats] = {}
        self._log_fn = log_fn

    def set_logger(self, log_fn: Callable[[str], None] | None) -> None:
        self._log_fn = log_fn

    def start(self) -> None:
        if self._cpu_profiler is not None:
            self._cpu_profiler.enable()

    def stop(self) -> None:
        if self._cpu_profiler is not None:
            self._cpu_profiler.disable()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            stats = self._sections.setdefault(name, ProfileSectionStats())
            stats.total_seconds += elapsed
            stats.calls += 1

    def report(self) -> None:
        if not self.enabled:
            return

        self._log_section_summary()
        self._log_cprofile_summary()
        self._dump_stats_file()

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)
            return
        print(message, flush=True)

    def _log_section_summary(self) -> None:
        if not self._sections:
            return

        total_profiled = sum(stats.total_seconds for stats in self._sections.values())
        self._log("Profiling summary by phase:")
        for name, stats in sorted(self._sections.items(), key=lambda item: item[1].total_seconds, reverse=True):
            avg_ms = 1000.0 * stats.total_seconds / max(stats.calls, 1)
            share = 100.0 * stats.total_seconds / max(total_profiled, 1e-12)
            self._log(
                f"  {name}: total={stats.total_seconds:.3f}s | calls={stats.calls} | avg={avg_ms:.2f}ms | share={share:.1f}%"
            )

    def _log_cprofile_summary(self) -> None:
        if self._cpu_profiler is None:
            return

        summary_stream = StringIO()
        stats = pstats.Stats(self._cpu_profiler, stream=summary_stream).sort_stats("cumulative")
        stats.print_stats(self.top_n)
        self._log("cProfile cumulative summary:")
        for line in summary_stream.getvalue().strip().splitlines():
            self._log(line)

    def _dump_stats_file(self) -> None:
        if self._cpu_profiler is None or self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._cpu_profiler.dump_stats(str(self.output_path))
        self._log(f"Profiler stats saved: {self.output_path}")
