from itertools import islice

from textual.binding import Binding
from textual.widgets import Log, Tabs

from logreader import LOG_LINE_LIMIT


class LogsPanel(Log):
    DEFAULT_CSS = "LogsPanel { height: 1fr; padding: 0 1; }"
    BINDINGS = [
        Binding("left", "previous_tab", show=False),
        Binding("right", "next_tab", show=False),
        Binding("end", "scroll_end", show=False),
    ]

    def __init__(self, reader):
        super().__init__(max_lines=LOG_LINE_LIMIT)
        self.reader = reader
        self.view = "log"
        self.buffer = None
        self.consumed_lines = 0
        self.option_width = 0

    def action_scroll_end(self):
        self.scroll_end(animate=False, immediate=True, x_axis=False)

    def action_previous_tab(self):
        tabs = self.app.query_one(Tabs)
        tabs.focus()
        tabs.action_previous_tab()

    def action_next_tab(self):
        tabs = self.app.query_one(Tabs)
        tabs.focus()
        tabs.action_next_tab()

    def update_data(self):
        if not self.size:
            return
        reader = self.reader
        if self.view == "log":
            buffer, line_count = reader.log_lines, reader.log_line_count
            self.max_lines = LOG_LINE_LIMIT
        else:
            buffer = reader.state.par_lines if self.view == "par" else reader.state.options
            line_count = len(buffer)
            self.max_lines = None
        reset = self.buffer is not buffer
        width = max((len(key) for key, _ in buffer), default=0) if self.view == "options" else 0
        reflow = self.view == "options" and width != self.option_width
        self.option_width = width
        scroll_y = self.scroll_y
        if reset or reflow:
            self.clear()
            self.buffer = buffer
            self.consumed_lines = 0
        added = line_count - self.consumed_lines
        if not added:
            return
        following = self.view == "log" and (reset or self.is_vertical_scroll_end)
        start = max(0, len(self.buffer) - added)
        dropped = (max(0, line_count - LOG_LINE_LIMIT) - max(0, self.consumed_lines - LOG_LINE_LIMIT)
                   if self.view == "log" else 0)
        lines = islice(self.buffer, start, None)
        if self.view == "options":
            lines = (f"{key:<{width}}  {value}" for key, value in lines)
        self.write_lines((line + "\n" for line in lines), scroll_end=following)
        self.consumed_lines = line_count
        if reset:
            scroll = self.scroll_end if self.view == "log" else self.scroll_home
            self.call_after_refresh(scroll, animate=False, x_axis=False)
        elif reflow:
            self.call_after_refresh(self.scroll_to, y=scroll_y, animate=False, immediate=True)
        elif not following:
            self.scroll_to(y=max(0, scroll_y - dropped), animate=False, immediate=True)
