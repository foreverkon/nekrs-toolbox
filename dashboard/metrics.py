from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Select, Static
from textual_plotext import PlotextPlot

from logreader import HISTORY_LIMIT
from plots import draw_plot


class MetricsPanel(VerticalScroll):
    DEFAULT_CSS = """
    MetricsPanel #metrics-controls { height: 3; margin-bottom: 1; }
    MetricsPanel Select { width: 1fr; }
    MetricsPanel #metrics-window { width: 24; margin-right: 1; }
    MetricsPanel Grid { height: 1fr; grid-rows: 1fr; grid-gutter: 1; }
    MetricsPanel PlotextPlot { height: 1fr; border: round #334155; }
    MetricsPanel #metrics-empty { height: auto; color: #94a3b8; }
    """

    class Update(Message, bubble=False):
        pass

    def __init__(self, reader, **kwargs):
        super().__init__(**kwargs)
        self.reader = reader
        self.groups = []
        self.charts = {}
        self.rendered = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="metrics-controls"):
            yield Select([(f"最近 {count} 步", count) for count in (100, 1000, HISTORY_LIMIT)],
                         value=1000, allow_blank=False, id="metrics-window")
            yield Select([], prompt="指标组", id="metrics-group")
        yield Static("暂无指标", id="metrics-empty")
        yield Grid(id="metrics-charts")

    def on_select_changed(self, event: Select.Changed):
        event.stop()
        if self.rendered is not None:
            self.update_data()

    def on_resize(self):
        self.update_layout()

    def update_layout(self):
        grid = self.query_one("#metrics-charts", Grid)
        columns = 1 if self.app.size.width < 100 else 2
        rows = max(1, (len(self.charts) + columns - 1) // columns)
        grid.styles.grid_size_columns = columns
        grid.styles.grid_size_rows = rows
        grid.styles.grid_columns = " ".join(["1fr"] * columns)
        grid.styles.min_height = rows * 10 + rows - 1

    def update_data(self):
        self.post_message(self.Update())

    async def on_metrics_panel_update(self):
        state = self.reader.state
        window = self.query_one("#metrics-window", Select).value
        selector = self.query_one("#metrics-group", Select)
        groups = list(state.metrics)
        if groups != self.groups:
            selected = selector.value
            self.groups = groups
            selector.set_options([(group, group) for group in groups])
            selector.value = selected if selected in groups else groups[0] if groups else Select.NULL
        selected = selector.value
        latest = max(state.step, state.metric_step)
        signature = (id(state), state.metric_revision, latest, window, selected)
        if signature == self.rendered:
            return
        fields = state.metrics.get(selected, {})
        grid = self.query_one("#metrics-charts", Grid)
        if list(fields) != list(self.charts):
            await grid.remove_children()
            self.charts = {name: PlotextPlot() for name in fields}
            if self.charts:
                await grid.mount(*self.charts.values())
            self.update_layout()
        grid.display = bool(self.charts)
        self.query_one("#metrics-empty").display = not self.charts
        for name, chart in self.charts.items():
            history = fields[name]
            steps = sorted(step for step in history if step >= latest - window + 1)
            draw_plot(chart, name, steps, [(None, [history[step] for step in steps])])
        self.rendered = signature
