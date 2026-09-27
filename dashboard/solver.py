from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, VerticalScroll
from textual.widgets import Select
from textual_plotext import PlotextPlot

from logreader import HISTORY_LIMIT
from plots import draw_plot


class SolverPanel(VerticalScroll):
    DEFAULT_CSS = """
    SolverPanel #solver-controls { height: 3; margin-bottom: 1; }
    SolverPanel Select { width: 1fr; }
    SolverPanel #solver-window { width: 24; margin-right: 1; }
    SolverPanel Grid { height: 1fr; min-height: 32; grid-size: 2 3; grid-columns: 1fr 1fr; grid-rows: 1fr; grid-gutter: 1; }
    SolverPanel PlotextPlot { height: 1fr; border: round #334155; }
    """

    def __init__(self, reader, **kwargs):
        super().__init__(**kwargs)
        self.reader = reader
        self.field_names = []
        self.rendered = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="solver-controls"):
            yield Select([(f"最近 {count} 步", count) for count in (100, 1000, HISTORY_LIMIT)],
                         value=1000, allow_blank=False, id="solver-window")
            yield Select([], prompt="求解场", id="solver-field")
        with Grid():
            for name in ("cfl", "dt", "iterations", "residuals", "divergence", "wall"):
                yield PlotextPlot(id=f"solver-{name}")

    def on_select_changed(self, event: Select.Changed):
        event.stop()
        if self.rendered is not None:
            self.update_data()

    def update_data(self):
        state = self.reader.state
        window = self.query_one("#solver-window", Select).value
        selector = self.query_one("#solver-field", Select)
        names = list(dict.fromkeys(name for sample in state.solver_samples for name in sample.fields))
        if names != self.field_names:
            selected = selector.value
            self.field_names = names
            selector.set_options([(name, name) for name in names])
            selector.value = selected if selected in names else names[0] if names else Select.NULL
        selected = selector.value
        signature = (id(state), state.solver_revision, window, selected)
        if signature == self.rendered:
            return
        self.rendered = signature
        samples = list(state.solver_samples)
        if samples:
            first_step = samples[-1].step - window + 1
            samples = [sample for sample in samples if sample.step >= first_step]
        steps = [sample.step for sample in samples]
        field_values = [sample.fields.get(selected, (None, None, None)) for sample in samples]
        self.draw("cfl", "CFL", steps, [("CFL", [sample.cfl for sample in samples])])
        self.draw("dt", "dt", steps, [("dt", [sample.dt for sample in samples])])
        self.draw("iterations", "迭代次数", steps, [(str(selected), [value[0] for value in field_values])])
        self.draw("residuals", "残差", steps, [
            ("resNorm0", [value[1] for value in field_values]),
            ("resNorm", [value[2] for value in field_values]),
        ], logarithmic=True)
        self.draw("divergence", "连续性误差", steps, [
            ("average", [sample.div_average for sample in samples]),
            ("L2", [sample.div_l2 for sample in samples]),
        ])
        self.draw("wall", "每步耗时 / s", steps, [("compute", [sample.wall for sample in samples])])

    def draw(self, name, title, steps, series, logarithmic=False):
        widget = self.query_one(f"#solver-{name}", PlotextPlot)
        draw_plot(widget, title, steps, series, logarithmic=logarithmic, integer=name == "iterations")
