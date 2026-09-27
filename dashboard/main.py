#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from time import monotonic

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, ProgressBar, Select, Static, TabbedContent, TabPane

from controls import SignalConfirmation, prepare_checkpoint, send_signal
from logreader import LogReader
from logs import LogsPanel
from metrics import MetricsPanel
from solver import SolverPanel


def number(value):
    return "—" if value is None else f"{value:g}"


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "—"
    if seconds < 1:
        return f"{seconds:.3f} s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return (f"{days}d " if days else "") + f"{hours:02}:{minutes:02}:{seconds:02}"


def date_time(value):
    return "—" if value is None else value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def fields(rows):
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(style="#94a3b8", width=16)
    table.add_column(ratio=1, overflow="fold")
    for label, value in rows:
        table.add_row(Text(label), Text(str(value) if value is not None and value != "" else "—"))
    return table


class Dashboard(App):
    TITLE = "nekRS Dashboard"
    BINDINGS = [("q", "quit", "退出"), ("p", "pause", "暂停 / 继续"), ("r", "reload", "重新读取")]
    CSS = """
    Screen { background: #0b1220; color: #e2e8f0; }
    Footer { background: #172235; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; }
    VerticalScroll { padding: 1 2; }
    .card { height: auto; border: round #334155; padding: 0 2; margin-bottom: 1; }
    .card Static { height: auto; }
    #operation { color: #67e8f9; text-style: bold; margin-bottom: 1; }
    #operation.failed { color: #fb7185; }
    #operation.finished { color: #4ade80; }
    #progress { width: 100%; margin: 1 0; }
    #progress Bar { width: 1fr; }
    #metadata, #live-values, #forecast { height: auto; }
    #metadata { margin-bottom: 1; }
    #case-info, #run-info { width: 1fr; }
    #case-info { margin-right: 1; }
    #progress-detail { width: 1fr; margin-right: 1; }
    #timing { width: 1fr; }
    #forecast { margin-bottom: 1; }
    #remaining { width: 1fr; }
    #finish { width: 2fr; }
    #remaining, #finish { color: #67e8f9; text-style: bold; }
    #source { color: #94a3b8; height: auto; }
    #job-controls { height: auto; margin-top: 1; }
    #job-controls Button { margin-right: 1; }
    Screen.compact #metadata, Screen.compact #live-values { layout: vertical; }
    Screen.compact #case-info, Screen.compact #run-info { width: 100%; }
    Screen.compact #case-info { margin-right: 0; }
    Screen.compact #progress-detail { width: 100%; margin-right: 0; }
    Screen.compact #timing { width: 100%; }
    Screen.compact SolverPanel Grid { height: auto; grid-size: 1 6; grid-columns: 1fr; grid-rows: 14; }
    """

    def __init__(self, path, refresh=0.5):
        super().__init__()
        self.reader = LogReader(path)
        self.refresh_seconds = refresh
        self.paused = False
        self.next_render = 0
        self.needs_render = True
        self.signaling = False

    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Overview", id="overview"):
                with VerticalScroll():
                    with Vertical(classes="card", id="case-card"):
                        with Horizontal(id="metadata"):
                            yield Static(id="case-info")
                            yield Static(id="run-info")
                        yield Static(id="paths")
                    with Vertical(classes="card", id="live-card"):
                        yield Static(id="operation", markup=False)
                        yield ProgressBar(total=None, show_eta=False, id="progress")
                        with Horizontal(id="forecast"):
                            yield Static(id="remaining", markup=False)
                            yield Static(id="finish", markup=False)
                        with Horizontal(id="live-values"):
                            yield Static(id="progress-detail")
                            yield Static(id="timing")
                        with Horizontal(id="job-controls"):
                            yield Button("保存重启场", id="checkpoint")
                            yield Button("结束计算", id="stop-job", variant="error")
                    yield Static(id="source", markup=False)
            with TabPane("Solver", id="solver"):
                yield SolverPanel(self.reader)
            with TabPane("Metrics", id="metrics"):
                yield MetricsPanel(self.reader)
            with TabPane("Logs", id="logs"):
                yield Select([("日志", "log"), ("PAR", "par"), ("Options", "options")],
                             value="log", allow_blank=False, id="log-view")
                yield LogsPanel(self.reader)
        yield Footer()

    def on_mount(self):
        for name, title in (("case", "算例信息"), ("live", "运行状态")):
            self.query_one(f"#{name}-card").border_title = title
        self.screen.set_class(self.size.width < 100, "compact")
        self.poll_log()

    def on_resize(self, event):
        self.default_screen.set_class(event.size.width < 100, "compact")

    def poll_log(self):
        if not self.paused:
            self.needs_render |= self.reader.read()
        now = monotonic()
        if self.needs_render and (self.reader.caught_up or not self.reader.exists or now >= self.next_render):
            self.update_overview()
            if self.query_one(TabbedContent).active == "solver":
                self.query_one(SolverPanel).update_data()
            elif self.query_one(TabbedContent).active == "metrics":
                self.query_one(MetricsPanel).update_data()
            elif self.query_one(TabbedContent).active == "logs":
                self.query_one(LogsPanel).update_data()
            self.needs_render = False
            self.next_render = now + self.refresh_seconds
        catching_up = self.reader.exists and not self.reader.caught_up and not self.paused
        self.poll_timer = self.set_timer(0.001 if catching_up else self.refresh_seconds, self.poll_log)

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated):
        if event.pane.id == "solver":
            self.query_one(SolverPanel).update_data()
        elif event.pane.id == "metrics":
            self.query_one(MetricsPanel).update_data()
        elif event.pane.id == "logs":
            panel = self.query_one(LogsPanel)
            self.call_after_refresh(panel.update_data)
            panel.focus()

    def on_select_changed(self, event: Select.Changed):
        if event.select.id == "log-view":
            event.stop()
            panel = self.query_one(LogsPanel)
            panel.view = event.value
            panel.update_data()

    def update_overview(self):
        reader, state = self.reader, self.reader.state
        self.update_controls()
        operation = self.query_one("#operation", Static)
        operation.update(state.operation + (f"  ·  {state.detail}" if state.detail else ""))
        operation.set_class(bool(state.result) and state.result != "正常结束", "failed")
        operation.set_class(state.result == "正常结束", "finished")
        target = "—"
        if state.stop_at == "numsteps" and state.num_steps is not None:
            target = f"{state.num_steps:,} 步"
        elif state.stop_at == "endtime" and state.end_time is not None:
            target = f"t = {number(state.end_time)}"
        if state.stop_at == "elapsedtime" and state.wall_limit is not None:
            target = f"运行耗时 {duration(state.wall_limit)}"
        progress = state.progress
        self.query_one("#progress", ProgressBar).update(total=100 if progress is not None else None,
                                                       progress=100 * progress if progress is not None else 0)
        self.query_one("#progress-detail", Static).update(fields([
            ("当前步", f"{state.step:,}"),
            ("物理时间", number(state.time if state.time is not None else state.start_time)),
            ("dt", number(state.dt)),
        ]))
        mode = "—" if state.variable_dt is None else ("自适应" if state.variable_dt else "固定")
        self.query_one("#run-info", Static).update(fields([
            ("运行开始", date_time(state.started_at)), ("起始物理时间", number(state.start_time)),
            ("终止条件", target), ("时间步模式", mode), ("初始化耗时", duration(state.initialization)),
        ]))
        self.query_one("#case-info", Static).update(fields([
            ("算例", state.case), ("作业 ID", state.job_id), ("nekRS", state.version),
            ("MPI 进程数", state.ranks), ("计算后端", state.backend),
            ("全局单元数", f"{state.elements:,}" if state.elements is not None else None),
            ("多项式阶数", state.order), ("标量场数量", state.scalars),
        ]))
        step_times = [sample.wall for sample in list(state.solver_samples)[-31:] if sample.wall is not None]
        per_step = sum(step_times) / len(step_times) if step_times else None
        self.query_one("#timing", Static).update(fields([
            ("累计耗时", duration(state.total_wall)),
            ("最近平均每步", f"{number(per_step)} s" if per_step is not None else None),
            ("最近一步耗时", f"{number(state.step_wall)} s" if state.step_wall is not None else None),
        ]))
        self.query_one("#remaining", Static).update("预计剩余  " + duration(state.remaining))
        self.query_one("#finish", Static).update("预计结束  " + date_time(state.expected_finish))
        self.query_one("#paths", Static).update(fields([
            ("运行目录", state.directory), ("日志路径", str(reader.path)),
        ]))
        if self.paused:
            status = "已暂停"
        elif not reader.exists:
            status = "等待日志文件创建"
        elif not reader.caught_up:
            status = "读取日志中"
        else:
            status = ""
        self.query_one("#source", Static).update(
            "  ·  ".join(part for part in (status, f"日志更新 {date_time(reader.modified)}") if part)
        )

    def update_controls(self):
        self.query_one("#checkpoint", Button).disabled = self.signaling
        self.query_one("#stop-job", Button).disabled = self.signaling

    def signal_unavailable(self, signal):
        reader, state = self.reader, self.reader.state
        if not reader.exists:
            return "日志文件不存在"
        if not reader.caught_up:
            return "日志正在读取，请稍后再试"
        if state.result:
            return f"计算已结束：{state.result}"
        if not state.job_id.isdecimal():
            return "日志中没有有效的 Slurm Job ID"
        if state.initialization is None:
            return "日志尚未显示初始化完成"
        if signal == "USR1" and not state.directory:
            return "日志中没有运行目录"
        return ""

    def on_button_pressed(self, event: Button.Pressed):
        job_id = self.reader.state.job_id
        if event.button.id == "checkpoint":
            signal = "USR1"
        elif event.button.id == "stop-job":
            signal = "USR2"
        else:
            return
        if reason := self.signal_unavailable(signal):
            self.notify(reason, severity="warning")
            return
        self.push_screen(SignalConfirmation(job_id, signal),
                         lambda confirmed: self.start_signal(job_id, signal) if confirmed else None)

    def start_signal(self, job_id, signal):
        if self.signaling:
            return
        if job_id != self.reader.state.job_id:
            self.notify("作业状态已变化，请重新操作", severity="warning")
            return
        if reason := self.signal_unavailable(signal):
            self.notify(reason, severity="warning")
            return
        self.signaling = True
        self.update_controls()
        self.run_worker(self.send_job_signal(job_id, signal, self.reader.state.directory))

    async def send_job_signal(self, job_id, signal, directory):
        try:
            if signal == "USR1":
                prepare_checkpoint(directory)
            await send_signal(job_id, signal)
        except (OSError, RuntimeError) as error:
            self.notify(str(error), title="发送失败", severity="error", timeout=8)
        else:
            self.notify(f"{signal} 已发送", title=f"作业 {job_id}")
        finally:
            self.signaling = False
            self.update_controls()

    def action_pause(self):
        self.poll_timer.stop()
        self.paused = not self.paused
        self.needs_render = True
        self.next_render = 0
        self.poll_log()

    def action_reload(self):
        self.poll_timer.stop()
        self.reader.reset()
        self.paused = False
        self.needs_render = True
        self.next_render = 0
        self.poll_log()


def main():
    parser = argparse.ArgumentParser(description="nekRS live log dashboard")
    parser.add_argument("log", type=Path, help="nekRS stdout / Slurm log path")
    parser.add_argument("--refresh", type=float, default=0.5, help="Refresh interval in seconds (default: 0.5)")
    args = parser.parse_args()
    if not math.isfinite(args.refresh) or args.refresh <= 0:
        parser.error("--refresh must be positive and finite")
    Dashboard(args.log, args.refresh).run()


if __name__ == "__main__":
    main()
