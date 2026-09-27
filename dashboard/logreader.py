from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|inf|nan)"
STEP = re.compile(rf"^step=\s*(\d+)\s+t=\s*({NUMBER})\s+dt=\s*({NUMBER})", re.I)
TIMING = re.compile(rf"^step=\s*(\d+)\s+elapsedStep=\s*({NUMBER})s\s+elapsedStepSum=\s*({NUMBER})s", re.I)
CFL = re.compile(rf"\bCFL=\s*({NUMBER})", re.I)
SOLVER = re.compile(rf"^step=\s*(\d+)\s+(.+?)\s*:\s*iter\s+(\d+)\s+resNorm0\s+({NUMBER})\s+resNorm\s+({NUMBER})", re.I)
DIVERGENCE = re.compile(rf"^step=\s*(\d+)\s+.*?divUErr\s*:\s*({NUMBER})\s+({NUMBER})", re.I)
IO_TIMING = re.compile(rf"^elapsed time:\s*({NUMBER})s", re.I)
METRIC = re.compile(rf"^METRIC\s+step=(\d+)\s+group=(\S+)((?:\s+[^\s=]+={NUMBER})*)$", re.I)
OPTION = re.compile(r"^key:\s*(.*?),\s*value:\s*(.*)$")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
HISTORY_LIMIT = 5000
LOG_LINE_LIMIT = 5000


@dataclass
class Sample:
    step: int
    time: float
    wall: float


@dataclass
class SolverSample:
    step: int
    dt: float | None = None
    cfl: float | None = None
    wall: float | None = None
    div_average: float | None = None
    div_l2: float | None = None
    fields: dict[str, tuple[int, float, float]] = field(default_factory=dict)


@dataclass
class RunState:
    case: str = ""
    directory: str = ""
    job_id: str = ""
    version: str = ""
    ranks: int | None = None
    backend: str = ""
    elements: int | None = None
    order: int | None = None
    scalars: int | None = None
    variable_dt: bool | None = None
    start_time: float | None = None
    end_time: float | None = None
    num_steps: int | None = None
    wall_limit: float | None = None
    stop_at: str = ""
    step: int = 0
    time: float | None = None
    dt: float | None = None
    started_at: datetime | None = None
    initialization: float | None = None
    elapsed: float | None = None
    step_wall: float | None = None
    total_wall: float | None = None
    operation: str = "等待日志"
    detail: str = ""
    result: str = ""
    samples: deque = field(default_factory=lambda: deque(maxlen=31))
    solver_samples: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))
    solver_revision: int = 0
    write_step: int | None = None
    write_wall: float = 0
    writing_field: bool = False
    metrics: dict = field(default_factory=dict)
    metric_step: int = 0
    metric_revision: int = 0
    par_lines: list[str] = field(default_factory=list)
    options: list[tuple[str, str]] = field(default_factory=list)

    def add_metrics(self, step, group, values):
        fields = self.metrics.setdefault(group, {})
        for name, value in values.items():
            history = fields.setdefault(name, {})
            history[step] = value
            if len(history) > HISTORY_LIMIT:
                del history[next(iter(history))]
        self.metric_step = max(self.metric_step, step)
        self.metric_revision += 1

    def solver_sample(self, step):
        if step != self.write_step:
            self.write_step, self.write_wall = step, 0
        if self.solver_samples and step < self.solver_samples[-1].step:
            self.solver_samples.clear()
        if not self.solver_samples or step != self.solver_samples[-1].step:
            self.solver_samples.append(SolverSample(step))
        self.solver_revision += 1
        return self.solver_samples[-1]

    @property
    def progress(self):
        if self.stop_at == "elapsedtime" and self.wall_limit is not None and self.total_wall is not None:
            return min(1.0, self.total_wall / self.wall_limit) if self.wall_limit > 0 else 1.0
        if self.stop_at == "numsteps" and self.num_steps is not None:
            return min(1.0, self.step / self.num_steps) if self.num_steps > 0 else 1.0
        if self.stop_at == "endtime" and self.end_time is not None and self.start_time is not None:
            current = self.start_time if self.time is None else self.time
            if not all(math.isfinite(value) for value in (current, self.start_time, self.end_time)):
                return None
            span = self.end_time - self.start_time
            return min(1.0, max(0.0, (current - self.start_time) / span)) if span > 0 else 1.0
        return None

    @property
    def rates(self):
        if len(self.samples) < 2:
            return None, None
        first, last = self.samples[0], self.samples[-1]
        wall = last.wall - first.wall
        steps = last.step - first.step
        simulation = last.time - first.time
        if not math.isfinite(wall) or wall <= 0 or steps <= 0:
            return None, None
        return wall / steps, wall / simulation if math.isfinite(simulation) and simulation > 0 else None

    @property
    def remaining(self):
        if self.result:
            return None
        if self.stop_at == "elapsedtime" and self.wall_limit is not None and self.total_wall is not None:
            return max(0, self.wall_limit - self.total_wall)
        per_step, per_time = self.rates
        if self.stop_at == "numsteps" and self.num_steps is not None and per_step is not None:
            return max(0, self.num_steps - self.samples[-1].step) * per_step
        if self.stop_at == "endtime" and self.end_time is not None and per_time is not None:
            return max(0, self.end_time - self.samples[-1].time) * per_time
        return None

    @property
    def expected_finish(self):
        remaining = self.remaining
        if remaining is None or self.started_at is None or self.total_wall is None:
            return None
        seconds = self.total_wall + remaining
        if not math.isfinite(seconds):
            return None
        return self.started_at + timedelta(seconds=seconds)


class LogReader:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.reset()

    def reset(self):
        self.state = RunState()
        self.log_lines = deque(maxlen=LOG_LINE_LIMIT)
        self.log_line_count = 0
        self.section = ""
        self.par_echo = None
        self.position = 0
        self.pending = b""
        self.tail = b""
        self.identity = None
        self.modified_ns = None
        self.modified = None
        self.exists = False
        self.caught_up = False

    def read(self):
        try:
            stat = self.path.stat()
            if ((stat.st_dev, stat.st_ino) == self.identity
                    and stat.st_size == self.position and stat.st_mtime_ns == self.modified_ns):
                changed = not self.exists
                self.exists = True
                return changed
            stream = self.path.open("rb")
        except FileNotFoundError:
            changed = self.exists
            self.exists = False
            return changed
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            rewritten = False
            if self.tail:
                stream.seek(self.position - len(self.tail))
                rewritten = stream.read(len(self.tail)) != self.tail
            if identity != self.identity or stat.st_size < self.position or rewritten:
                self.reset()
                self.identity = identity
            self.exists = True
            self.modified_ns = stat.st_mtime_ns
            self.modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            stream.seek(self.position)
            chunk = stream.read(1024 * 1024)
            self.tail = (self.tail + chunk)[-256:]
            self.position = stream.tell()
            self.caught_up = self.position >= stat.st_size
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            self.feed(line.decode("utf-8", errors="replace"))
        return True

    def feed(self, line):
        if "\x1b" in line:
            line = ANSI.sub("", line)
        self.log_lines.append(line.removesuffix("\r"))
        self.log_line_count += 1
        raw_line = line.removesuffix("\r")
        line = line.strip()
        if self.par_echo and not line.startswith("<<<"):
            self.par_echo = False
        if not line or line == "::":
            return
        state = self.state
        if line.startswith("METRIC "):
            if match := METRIC.fullmatch(line):
                values = dict(token.split("=", 1) for token in match[3].split())
                state.add_metrics(int(match[1]), match[2], {name: float(value) for name, value in values.items()})
            return
        if line.startswith("step="):
            if match := STEP.match(line):
                state.step, state.time, state.dt = int(match[1]), float(match[2]), float(match[3])
                sample = state.solver_sample(state.step)
                sample.dt = state.dt
                if match := CFL.search(line):
                    sample.cfl = float(match[1])
                state.operation, state.detail = "推进求解", ""
                return
            elif match := TIMING.match(line):
                step, elapsed = int(match[1]), float(match[3])
                sample = state.solver_sample(step)
                state.step_wall = max(0.0, float(match[2]) - state.write_wall)
                sample.wall = state.step_wall
                state.write_step, state.write_wall = step + 1, 0
                if state.initialization is not None:
                    state.total_wall = state.initialization + elapsed
                state.elapsed = elapsed
                if int(match[1]) == state.step and state.time is not None:
                    if state.samples and state.step <= state.samples[-1].step:
                        state.samples.clear()
                    if not state.samples and state.start_time is not None:
                        state.samples.append(Sample(0, state.start_time, 0))
                    state.samples.append(Sample(state.step, state.time, elapsed))
                return
            elif "resNorm" in line:
                if match := SOLVER.match(line):
                    name = " ".join(match[2].split())
                    state.solver_sample(int(match[1])).fields[name] = (int(match[3]), float(match[4]), float(match[5]))
                return
            elif "divUErr" in line:
                if match := DIVERGENCE.match(line):
                    sample = state.solver_sample(int(match[1]))
                    sample.div_average, sample.div_l2 = float(match[2]), float(match[3])
                return
        new_case = re.match(r"reading (.+\.par)$", line)
        if (new_case or line.startswith(("UTC time:", "JOB_ID="))) and state.started_at is not None:
            self.state = state = RunState()
            self.section = ""
            self.par_echo = None

        if new_case:
            path = Path(new_case[1])
            state.case = path.stem
            if path.is_absolute():
                state.directory = str(path.parent)
            state.operation = "读取算例配置"
        elif line.startswith("CURRENT_DIR="):
            state.directory = line.partition("=")[2]
        elif line.startswith("JOB_ID="):
            state.job_id = line.partition("=")[2]
        elif line.startswith("UTC time:"):
            state.started_at = datetime.strptime(line.partition(":")[2].strip(), "%a %b %d %H:%M:%S %Y").replace(tzinfo=timezone.utc)
        elif match := re.match(r"v\d+\.\S+.*", line):
            state.version = match[0]
        elif match := re.match(r"MPI tasks:\s*(\d+)", line):
            state.ranks = int(match[1])
        elif line.startswith("active occa mode:"):
            state.backend = line.partition(":")[2].strip()

        if line.startswith("<<<"):
            if self.par_echo is None:
                self.par_echo = True
            if self.par_echo:
                state.par_lines.append(raw_line.lstrip()[3:].removeprefix(" "))
            echoed = line[3:].strip()
            if match := re.fullmatch(r"\[([^]]+)\]", echoed):
                self.section = match[1].lower()
            elif self.section == "general" and (match := re.match(r"(\w+)\s*=\s*([^#;]+)", echoed)):
                self.parameter(match[1].lower(), match[2].strip())
            return

        if match := OPTION.match(line):
            key, value = match[1].strip(), match[2].strip()
            state.options.append((key, value))
            numeric = {
                "START TIME": "start_time", "END TIME": "end_time", "DT": "dt",
                "NUMBER TIMESTEPS": "num_steps", "POLYNOMIAL DEGREE": "order",
                "NUMBER OF SCALARS": "scalars",
            }
            if key in numeric:
                converter = int if key in ("NUMBER TIMESTEPS", "POLYNOMIAL DEGREE", "NUMBER OF SCALARS") else float
                setattr(state, numeric[key], converter(value))
            elif key == "VARIABLE DT":
                state.variable_dt = value == "TRUE"
            elif key == "STOP AT ELAPSED TIME":
                state.wall_limit = 60 * float(value)
                state.stop_at = "elapsedtime"
            elif key == "CASENAME":
                state.case = value
            elif key == "THREAD MODEL":
                state.backend = value
            return

        if match := re.search(rf"initialization took\s+({NUMBER})\s*s", line):
            state.initialization = state.total_wall = float(match[1])
            state.operation = "初始化完成"
        elif match := re.search(rf"totalElapsed=\s*({NUMBER})s", line):
            state.total_wall = float(match[1])
        elif match := re.match(rf"timestepping to time\s+({NUMBER})", line):
            state.end_time = float(match[1])
            if state.stop_at != "elapsedtime":
                state.stop_at = "endtime"
            state.write_step, state.write_wall = 1, 0
            state.operation = "推进求解"
        elif match := re.match(r"timestepping for\s+(\d+)\s+steps", line):
            state.num_steps = int(match[1])
            if state.stop_at != "elapsedtime":
                state.stop_at = "numsteps"
            state.write_step, state.write_wall = 1, 0
            state.operation = "推进求解"
        elif "skip timestepping" in line:
            state.operation = "跳过时间推进"
        elif match := re.search(r"finished with exit code\s+(-?\d+)", line):
            state.result = "正常结束" if int(match[1]) == 0 else f"异常退出（退出码 {match[1]}）"
            state.operation, state.detail = state.result, ""
        elif re.search(r"FATAL ERROR|MPI_ABORT|Segmentation fault|DUE TO TIME LIMIT|CANCELLED", line, re.I):
            state.result = state.operation = "异常终止"
            state.detail = line

        if state.result:
            return
        if line.startswith("writing to field file"):
            state.writing_field = True
        elif line.startswith("reading checkpoint"):
            state.writing_field = False
        elif match := IO_TIMING.match(line):
            if state.writing_field and state.write_step is not None:
                state.write_wall += float(match[1])
            state.writing_field = False
        operations = (
            ("Initializing device", "初始化计算设备"),
            ("building nekInterface", "编译 nekInterface"),
            ("loading nek ...", "加载 nek 内核"),
            ("building udf", "编译 UDF"),
            ("JIT compiling kernels", "编译计算内核"),
            ("loading mesh from nek", "读取网格"),
            ("reading checkpoint", "读取重启场"),
            ("writing to field file", "写出场数据"),
            ("autotuning", "自动调优计算内核"),
            ("INITIAL CONDITION", "设置初始条件"),
        )
        for marker, operation in operations:
            if marker in line:
                state.operation, state.detail = operation, ""
                break
        if match := re.search(r"Nelements:\s*(\d+)", line):
            state.elements = int(match[1])
        if line.startswith("fileName:"):
            state.detail = line.partition(":")[2].strip()
        if line.startswith("elapsed time:") and state.operation in ("读取重启场", "写出场数据"):
            state.operation += "完成"

    def parameter(self, key, value):
        state = self.state
        if key == "stopat":
            state.stop_at = value.lower()
        elif key in ("starttime", "endtime", "numsteps", "polynomialorder"):
            name = {"starttime": "start_time", "endtime": "end_time", "numsteps": "num_steps", "polynomialorder": "order"}[key]
            setattr(state, name, int(value) if key in ("numsteps", "polynomialorder") else float(value))
        elif key == "dt":
            state.variable_dt = "targetcfl" in value.lower()
            if not state.variable_dt and re.fullmatch(NUMBER, value, re.I):
                state.dt = float(value)
        elif key == "variabledt":
            state.variable_dt = value.lower() in ("true", "yes")
        elif key == "elapsedtime":
            state.wall_limit = 60 * float(value)
