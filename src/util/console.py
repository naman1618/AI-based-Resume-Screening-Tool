from rich.console import Console
import rich.terminal_theme
import builtins

console = Console(record=True)


def custom_print(*params, flush: bool = False, **kwargs):
    console.print(*params, **kwargs)
    console.save_html(
        "output.html", theme=rich.terminal_theme.DIMMED_MONOKAI, clear=False
    )


builtins.print = custom_print
