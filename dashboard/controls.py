import asyncio
from pathlib import Path
import re

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


def prepare_checkpoint(directory):
    path = Path(directory) / "nekrs.upd"
    lines = path.read_text().splitlines(keepends=True) if path.exists() else []
    found = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("["):
            break
        if re.match(r"\s*checkpoint\s*=", line, re.I):
            lines[index] = "checkpoint = true\n"
            found = True
    if not found:
        lines.insert(0, "checkpoint = true\n")
    path.write_text("".join(lines))


async def send_signal(job_id, signal):
    process = await asyncio.create_subprocess_exec(
        "scancel", "--batch", f"--signal={signal}", "--", job_id,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        message = (stderr or stdout).decode(errors="replace").strip()
        raise RuntimeError(message or f"scancel 退出码 {process.returncode}")


class SignalConfirmation(ModalScreen[bool]):
    BINDINGS = [("escape", "cancel", "取消")]
    DEFAULT_CSS = """
    SignalConfirmation { align: center middle; background: $background 70%; }
    SignalConfirmation > Vertical { width: 52; max-width: 95%; height: auto; padding: 1 2; border: round #334155; background: #172235; }
    SignalConfirmation Static { height: auto; margin-bottom: 1; }
    SignalConfirmation Horizontal { height: auto; align-horizontal: right; }
    SignalConfirmation Button { margin-left: 1; }
    """

    def __init__(self, job_id, signal):
        super().__init__()
        self.job_id = job_id
        self.signal = signal

    def compose(self) -> ComposeResult:
        with Vertical():
            action = "保存重启场" if self.signal == "USR1" else "结束计算"
            yield Static(f"{action}？\n作业 {self.job_id}", markup=False)
            with Horizontal():
                yield Button("取消", id="cancel-signal")
                yield Button("确认", variant="primary" if self.signal == "USR1" else "error", id="confirm-signal")

    def on_mount(self):
        self.query_one("#cancel-signal", Button).focus()

    def on_button_pressed(self, event: Button.Pressed):
        event.stop()
        self.dismiss(event.button.id == "confirm-signal")

    def action_cancel(self):
        self.dismiss(False)
