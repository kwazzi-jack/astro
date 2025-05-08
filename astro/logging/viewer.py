from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import pathlib
from typing import Any

import click
import dateutil.parser
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, HorizontalGroup, Horizontal
from textual.reactive import var
from textual.screen import Screen
from textual.widgets import (
    Footer,
    Header,
    Input,
    DataTable,
    Static,
    Select,
    Button,
    RichLog,
)

from astro.paths import get_module_dir
from astro.logging.base import _LOG_FILE

# CSS path for Textual
_TEXTUAL_CSS_PATH = get_module_dir(__file__) / "textual.css"
if not _TEXTUAL_CSS_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find `{_TEXTUAL_CSS_PATH.name}` at `{_TEXTUAL_CSS_PATH}`"
    )

# Colors for each log level
_LEVEL_COLORS = {
    "DEBUG": "bold green",
    "INFO": "bold cyan",
    "WARNING": "bold yellow",
    "ERROR": "bold orange",
    "CRITICAL": "bold red",
}


# --- Helper Functions ---
def parse_timestamp(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return dateutil.parser.parse(ts)
        except Exception:
            return datetime.max.replace(tzinfo=timezone.utc)


def format_datetime(dt: datetime | None) -> str:
    template = "[underline white]{date}[/] [dim yellow]{time}+{tz}[/]"
    if dt is None:
        return template.format(date="????-??-??", time="??:??:??.???", tz="??:??")

    ms_str = f".{dt.microsecond // 1000:03d}"
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M:%S") + ms_str

    tz = dt.strftime("%z")
    if tz:
        if tz[0] == "+":
            tz_str = f"{tz[1:3]}:{tz[3:5]}"
        else:
            tz_str = f"{tz[:3]}:{tz[3:5]}"
    else:
        tz_str = "00:00"

    return template.format(date=date_str, time=time_str, tz=tz_str)


# --- Popup Screen ---
class LogDetailPopup(Screen):
    def __init__(self, entry: dict, log_list: "LogList", current_index: int, **kwargs):
        super().__init__(**kwargs)
        self.entry = entry
        self.log_name = entry.get("name", "Unknown")
        self.level = entry.get("level", "UNKNOWN")
        self.log_list = log_list
        self.current_index = current_index

    def compose(self) -> ComposeResult:
        level_style = _LEVEL_COLORS.get(self.level.upper(), "white")
        yield Static(
            " ".join(
                [
                    f"[bold white]#{self.current_index:08d}[/]",
                    f"[underline green]{self.log_name}[/]",
                    f"[{level_style}]{self.level}[/]",
                ]
            ),
            classes="popup-title",
        )
        yield RichLog(id="detail-richlog", classes="detail-content", auto_scroll=False)
        yield Static(Text("PREV LOG ← | NEXT LOG →", style="dim italic"))
        yield Button("Close", classes="close-button")

    def on_mount(self):
        richlog = self.query_one("#detail-richlog", RichLog)
        self._format_richlog_entry(richlog)
        richlog.focus()

    def _format_richlog_entry(self, richlog: RichLog):
        """Format and display a log entry with rich styling."""
        header_style = "bold underline yellow"
        key_style = "yellow"
        value_style = "white"
        indent = 2

        def recursive_write(
            content: dict[str, Any] | list[Any], depth: int = 0, max_depth: int = 6
        ):
            INDENT = " " * (indent * depth)

            if isinstance(content, dict):
                for key, value in content.items():
                    if value is None:
                        richlog.write(
                            Text(f"{INDENT}{key.capitalize()}: ", style=key_style)
                            + Text("None", style=value_style)
                        )
                    elif isinstance(value, (dict, list)) and len(value) > 0:
                        richlog.write(
                            Text(f"{INDENT}{key.capitalize()}:", style=key_style)
                        )
                        recursive_write(
                            value, depth + 1 if depth < max_depth else depth
                        )
                    else:
                        # Handle primitive values
                        richlog.write(
                            Text(f"{INDENT}{key.capitalize()}: ", style=key_style)
                            + Text(str(value), style=value_style)
                        )
            elif isinstance(content, list):
                for i, item in enumerate(content):
                    list_prefix = f"{INDENT}[{i}] "

                    if isinstance(item, (dict, list)) and len(item) > 0:
                        richlog.write(Text(list_prefix, style=key_style))
                        recursive_write(item, depth + 1 if depth < max_depth else depth)
                    else:
                        # Handle primitive values in list
                        richlog.write(
                            Text(list_prefix, style=key_style)
                            + Text(str(item), style=value_style)
                        )
            else:
                # Handle unexpected content types
                richlog.write(Text(f"{INDENT}{str(content)}", style=value_style))

        entry = deepcopy(self.entry)

        richlog.write(Text("Primary:", style=header_style))
        richlog.write(
            Text("Log Level: ", style=key_style)
            + Text(f"{entry.pop('level', 'None')}", style=value_style)
        )
        richlog.write(
            Text("Timestamp: ", style=key_style)
            + Text(f"{entry.pop('timestamp', 'None')}", style=value_style)
        )
        richlog.write(
            Text("Name: ", style=key_style)
            + Text(f"{entry.pop('name', 'None')}", style=value_style)
        )
        richlog.write(
            Text("Filename: ", style=key_style)
            + Text(f"{entry.pop('filename', 'None')}", style=value_style)
        )
        richlog.write(
            Text("Lineno: ", style=key_style)
            + Text(f"{entry.pop('lineno', 'None')}", style=value_style)
        )
        richlog.write(
            Text("Message: ", style=key_style)
            + Text(f"{entry.pop('message', 'None')}", style=value_style)
        )

        exception = entry.pop("exception", None)

        # Exception section
        if exception:
            richlog.write(Text("Exception:", style=header_style))
            richlog.write(Text(exception, style=value_style))

        if entry:
            richlog.write(Text("Additional Fields:", style=header_style))
            recursive_write(entry)

    def on_button_pressed(self, event: Button.Pressed):
        self.app.pop_screen()

    def on_key(self, event):
        if event.key == "right":
            display_indices = self.app.query_one(LogList).display_indices
            current_i = display_indices.index(self.current_index)
            if self.current_index < len(self.log_list.entries) - 1 and current_i < len(
                display_indices
            ):
                next_index = display_indices[current_i + 1]
                new_popup = LogDetailPopup(
                    self.log_list.entries[next_index], self.log_list, next_index
                )
                self.app.pop_screen()
                self.app.push_screen(new_popup)
        elif event.key == "left":
            display_indices = self.app.query_one(LogList).display_indices
            current_i = display_indices.index(self.current_index)
            if self.current_index > 0 and current_i > 0:
                prev_index = display_indices[current_i - 1]
                new_popup = LogDetailPopup(
                    self.log_list.entries[prev_index], self.log_list, prev_index
                )
                self.app.pop_screen()
                self.app.push_screen(new_popup)


# --- Widgets ---
class StatusBar(Static):
    def update_stats(
        self, total_logs: int, displayed_logs: int, level_counts: dict[str, int]
    ):
        level_stats = " | ".join(
            [
                f"[{_LEVEL_COLORS[level.upper()]}]{level}:[/] {count}"
                for level, count in level_counts.items()
            ]
        )
        self.update(
            f"Showing [yellow]{displayed_logs}[/] of [yellow]{total_logs}[/] logs | {level_stats}"
        )


class FilterBar(HorizontalGroup):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Filter by message...", id="message-filter")
        yield Select(
            [
                ("ALL", "ALL"),
                ("DEBUG", "DEBUG"),
                ("INFO", "INFO"),
                ("WARNING", "WARNING"),
                ("ERROR", "ERROR"),
                ("CRITICAL", "CRITICAL"),
            ],
            value="ALL",
            id="level-filter",
        )
        with Horizontal(classes="button-group"):
            yield Button("Clear", id="clear-filters")
            yield Button("Reload", id="reload-logs")


def log_timestamp_key(entry: dict) -> float:
    ts = parse_timestamp(entry.get("timestamp", None))
    if ts is None:
        return 0
    else:
        return ts.timestamp()


class LogList(DataTable):

    def __init__(self, logfile_path: pathlib.Path, **kwargs):
        super().__init__(**kwargs)
        self.logfile_path = logfile_path
        self.entries = []  # All log entries, unfiltered
        self.zebra_stripes = True
        self.display_indices = []  # Indices into self.entries for current filter
        self.loading = False
        self._loading_indicator = None

    def _log_entry_generator(self):
        """Yield log entries from the file one by one."""
        try:
            with self.logfile_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.app.notify(f"Error reading logs: {e}")

    def load_logs(self):
        self.entries = []
        self.loading = True
        self.display_indices = []
        if self._loading_indicator is None:
            self._loading_indicator = Static(
                "[yellow]Loading logs...[/]", id="loading-indicator"
            )
            self.parent.mount(self._loading_indicator, before=self)
        else:
            self._loading_indicator.display = True
        try:
            # Load all entries into self.entries (unfiltered)
            for entry in self._log_entry_generator():
                self.entries.append(entry)
            # Sort all entries by timestamp (descending)
            self.entries.sort(
                key=log_timestamp_key,
                reverse=True,
            )
            # After loading, apply current filters
            self.apply_filters(
                self.app.message_filter, self.app.level_filter, refresh_table=True
            )
        except Exception as e:
            self.entries = []
            self.clear()
            self.app.notify(f"Error loading logs: {e}")
        finally:
            self.loading = False
            if self._loading_indicator:
                self._loading_indicator.display = False

    def refresh_view(self):
        self.clear(columns=True)
        self.add_columns("TIMESTAMP", "LEVEL", "NAME", "MESSAGE")
        for idx in self.display_indices:
            entry = self.entries[idx]
            dt = parse_timestamp(entry.get("timestamp", ""))
            level = entry.get("level", "UNKNOWN")
            level_text = Text(level, style=_LEVEL_COLORS.get(level.upper(), "white"))
            name_text = Text(entry.get("name", "Unknown"), style="underline green")
            message_text = Text(
                entry.get("message", ""),
                style=_LEVEL_COLORS.get(level.upper(), "white"),
            )
            label = Text(str(idx), style="bold yellow")  # Use original index as label
            self.add_row(
                format_datetime(dt),
                level_text,
                name_text,
                message_text,
                key=str(idx),
                label=label,
            )

    def on_mount(self):
        self.load_logs()

    def apply_filters(
        self, message_filter: str, level_filter: str, refresh_table: bool = True
    ):
        if self.loading:
            return
        # If entries not loaded yet, do nothing
        if not self.entries:
            return
        message_filter = (message_filter or "").lower()
        level_filter = (level_filter or "ALL").upper()
        all_level = level_filter == "ALL"
        self.display_indices = [
            idx
            for idx, entry in enumerate(self.entries)
            if (
                (
                    message_filter in entry.get("message", "").lower()
                    if message_filter
                    else True
                )
                and (all_level or entry.get("level", "").upper() == level_filter)
            )
        ]
        if refresh_table:
            self.refresh_view()


class LogViewerApp(App):
    CSS_PATH = str(_TEXTUAL_CSS_PATH)
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("r", "reload_logs", "Reload logs"),
        ("f", "focus_filter", "Focus filter"),
        ("escape", "dismiss_popup", "Close popup"),
        ("ctrl+c", "quit", "Quit"),
    ]

    message_filter = var("")
    level_filter = var("")

    def __init__(self, logfile_path: pathlib.Path, **kwargs):
        super().__init__(**kwargs)
        self.logfile_path = logfile_path
        self.is_resetting = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield FilterBar(id="filter-bar")
        with Container(id="main-content"):
            yield LogList(self.logfile_path, id="log-list")
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self):
        self.query_one(LogList).focus()
        self.update_stats()

    def on_data_table_cell_selected(self, event: DataTable.CellSelected):
        if event.cell_key is not None:
            row_key = event.cell_key[0].value
            try:
                index = int(row_key)
                log_list = self.query_one(LogList)
                entry = log_list.entries[index]
                popup = LogDetailPopup(entry, log_list, index)
                self.push_screen(popup)
            except (ValueError, IndexError):
                self.notify("Invalid selection", severity="error")

    def action_focus_filter(self):
        self.query_one("#message-filter", Input).focus()

    def action_reload_logs(self):
        self.query_one(LogList).load_logs()
        self.update_stats()
        self.notify("Logs reloaded")

    def update_stats(self):
        log_list = self.query_one(LogList)
        total_logs = len(log_list.entries)
        displayed_logs = len(log_list.display_indices)
        level_counts = defaultdict(int)

        for idx in log_list.display_indices:
            level = log_list.entries[idx].get("level", "UNKNOWN").upper()
            level_counts[level] += 1

        status_bar = self.query_one(StatusBar)
        status_bar.update_stats(total_logs, displayed_logs, level_counts)

    def action_dismiss_popup(self):
        if len(self.screen_stack) > 1:
            self.pop_screen()

    def on_input_changed(self, event: Input.Changed):
        if not self.is_resetting and event.input.id == "message-filter":
            self.message_filter = event.value

    def on_select_changed(self, event: Select.Changed):
        if not self.is_resetting and event.select.id == "level-filter":
            if event.value == Select.BLANK:
                self.level_filter = ""
                self.query_one(Select).value = "ALL"
            else:
                self.level_filter = event.value

    def watch_message_filter(self, message_filter: str):
        log_list = self.query_one(LogList)
        log_list.apply_filters(message_filter, self.level_filter)
        self.update_stats()

    def watch_level_filter(self, level_filter: str):
        log_list = self.query_one(LogList)
        log_list.apply_filters(self.message_filter, level_filter)
        self.update_stats()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear-filters":
            self.is_resetting = True
            # Reset filters
            self.message_filter = ""
            self.level_filter = ""
            # Clear UI elements
            self.query_one("#message-filter", Input).clear()
            self.query_one("#level-filter", Select).value = "ALL"
            self.is_resetting = False
        elif event.button.id == "reload-logs":
            self.action_reload_logs()


@click.command()
@click.argument("logfile_path", type=click.Path(path_type=pathlib.Path), required=False)
def view_logs(logfile_path):
    logfile = logfile_path or _LOG_FILE
    app = LogViewerApp(logfile_path=logfile)
    app.run()


if __name__ == "__main__":
    view_logs()
