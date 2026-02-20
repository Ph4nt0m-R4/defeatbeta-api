import logging
import time
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Configure Custom Flow Theme
custom_theme = Theme({
    "step": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "subtle": "dim white",
    "key": "bold blue",
    "value": "default"
})

console = Console(theme=custom_theme)

class FlowChartLogger:
    """
    Renders logs as a detailed Top-Down (TD) Flowchart with embedded stats.
    """
    def __init__(self):
        self.first_step = True
        self._start_time = None

    def start_timer(self):
        self._start_time = time.monotonic()
        self.first_step = True  # Reset for new flow

    def get_duration(self) -> str:
        if self._start_time is None:
            return ""
        elapsed = time.monotonic() - self._start_time
        return f"{elapsed:.4f}s"

    def node(self, title: str, details: dict = None, style: str = "step", subtitle: str = ""):
        """
        Draws a flow node with an optional table of details inside.
        """
        # Draw arrow if not first step
        if not self.first_step:
            console.print(f"   [dim]│[/]")
            console.print(f"   [dim]▼[/]")
        
        # Build Content
        items = [Text(title, style="bold")]
        
        if details:
            # Create a grid for key-value pairs
            grid = Table.grid(padding=(0, 2))
            grid.add_column(style="key", justify="right")
            grid.add_column(style="value", justify="left")
            
            for k, v in details.items():
                grid.add_row(f"{k}:", str(v))
            
            items.append(Text("──────", style="dim")) # Separator
            items.append(grid)

        # Create Panel
        panel = Panel(
            Group(*items), 
            style=style, 
            subtitle=subtitle, 
            expand=False,
            padding=(0, 2)
        )
        console.print(panel)
        self.first_step = False

    def end(self, message: str, details: dict = None):
        self.node(message, details=details, style="success", subtitle="[Done]")

# Instantiate global loggers
flow_log = FlowChartLogger()
logger = logging.getLogger("SmartParquet")
