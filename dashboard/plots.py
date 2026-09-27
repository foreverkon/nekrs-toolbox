import math


def draw_plot(widget, title, steps, series, logarithmic=False, integer=False):
    widget.border_title = title
    plot = widget.plt
    plot.clear_figure()
    plot.xlabel("step")
    plot.grid(True, True)
    if steps:
        if len(steps) == 1:
            plot.xlim(steps[0] - 0.5, steps[0] + 0.5)
        else:
            plot.xlim(steps[0], steps[-1])
        ticks = sorted({steps[round(index * (len(steps) - 1) / 3)] for index in range(4)})
        plot.xticks(ticks, [str(step) for step in ticks])
    values = [value for _, data in series for value in data if value is not None and math.isfinite(value)]
    widget.border_subtitle = ""
    if values and not logarithmic:
        lower, upper = min(values), max(values)
        ticks = [lower + index * (upper - lower) / 4 for index in range(5)]
        ticks = sorted(set(round(tick) if integer else tick for tick in ticks))
        for precision in range(3, 18):
            labels = [f"{tick:.{precision}g}" for tick in ticks]
            if len(set(labels)) == len(ticks):
                break
        plot.yticks(ticks, labels)
    if logarithmic:
        positive = [math.log10(value) for value in values if value > 0]
        lower = math.floor(min(positive)) if positive else 0
        upper = math.ceil(max(positive)) if positive else 0
        zero_level = lower - 1
        ticks = sorted({round(lower + index * (upper - lower) / 3) for index in range(4)}) if positive else []
        labels = [f"1e{tick}" for tick in ticks]
        if 0 in values:
            ticks.insert(0, zero_level)
            labels.insert(0, "0")
            widget.border_subtitle = "0 单独列示"
        plot.yticks(ticks, labels)
        plot.ylim(zero_level - 0.3 if 0 in values else lower - 0.3, upper + 0.3)
    plotted = False
    for (label, data), color in zip(series, ("cyan", "yellow")):
        segment_steps, segment_values, zero_steps = [], [], []
        labeled = False
        for step, value in zip(steps + [None], data + [None]):
            valid = value is not None and math.isfinite(value) and (not logarithmic or value > 0)
            if valid:
                segment_steps.append(step)
                segment_values.append(math.log10(value) if logarithmic else value)
            else:
                if logarithmic and value == 0:
                    zero_steps.append(step)
                if segment_steps:
                    plot.plot(segment_steps, segment_values, color=color, marker="braille",
                              label=label if not labeled else None)
                    plotted = labeled = True
                    segment_steps, segment_values = [], []
        if zero_steps:
            plot.scatter(zero_steps, [zero_level] * len(zero_steps), color=color,
                         marker="braille", label=label if not labeled else None)
            plotted = True
    if not plotted:
        widget.border_subtitle = "暂无数据"
    widget.refresh()
