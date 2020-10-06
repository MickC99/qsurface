from .sim import PerfectMeasurements as SimPM, FaultyMeasurements as SimFM
from .._template.plot import PerfectMeasurements as TemplatePPM, CodePlotPM as TemplateCPPM


class CodePlotPM(TemplateCPPM):
    """Toric code plot for perfect measurements."""

    def __init__(self, code, *args, **kwargs) -> None:
        self.main_boundary = [-0.25, -0.25, code.size[0] + 0.5, code.size[1] + 0.5]
        self.legend_loc = (1.3, 0.95)
        super().__init__(code, *args, **kwargs)

    def parse_boundary_coordinates(self, size, *args):
        options = {-1: [*args]}
        for i, arg in enumerate(args):
            if arg == 0:
                options[i] = [*args]
                options[i][i] = size
        diff = {
            option: sum([abs(args[i] - args[j]) for i in range(len(args)) for j in range(i + 1, len(args))])
            for option, args in options.items()
        }
        return options[min(diff, key=diff.get)]

class PerfectMeasurements(SimPM, TemplatePPM):
    """Plotting toric code class for perfect measurements."""
    FigureClass = CodePlotPM
