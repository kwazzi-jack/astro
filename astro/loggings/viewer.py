import json
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timezone

import click
import dateutil.parser
import pandas as pd
import streamlit as st
from streamlit import runtime
from streamlit.web import cli as stcli

from astro.loggings.base import _get_log_file, get_logger
from astro.paths import find_latest_log_file, get_available_log_files, get_module_dir

# Load logger
_logger = get_logger(__file__)


# Log level colors (matching original textual viewer)
LEVEL_COLORS = {
    "DEBUG": "#22c55e",  # green
    "INFO": "#06b6d4",  # cyan
    "WARNING": "#eab308",  # yellow
    "ERROR": "#f97316",  # orange
    "CRITICAL": "#dc2626",  # red
}

# CSS for styling
CUSTOM_CSS = """
<style>
    .log-level-debug { color: #22c55e; font-weight: bold; }
    .log-level-info { color: #06b6d4; font-weight: bold; }
    .log-level-warning { color: #eab308; font-weight: bold; }
    .log-level-error { color: #f97316; font-weight: bold; }
    .log-level-critical { color: #dc2626; font-weight: bold; }

    .log-message { font-family: monospace; }
    .log-timestamp { color: #6b7280; font-size: 0.9em; }
    .log-name { color: #10b981; font-weight: 500; }

    .metric-container {
        background: #1f2937;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-label { color: #9ca3af; font-size: 0.8em; }
    .metric-value { color: #f3f4f6; font-size: 1.2em; font-weight: bold; }

    .stExpander > details > summary {
        background-color: #374151;
        border-radius: 0.25rem;
        padding: 0.5rem;
    }
</style>
"""


def parse_timestamp(timestamp_str: str) -> datetime | None:
    """Parse timestamp string to datetime object."""
    try:
        return datetime.fromisoformat(timestamp_str)
    except Exception:
        try:
            return dateutil.parser.parse(timestamp_str)
        except Exception:
            return datetime.max.replace(tzinfo=timezone.utc)


def format_datetime_for_display(dt: datetime | None, format_type: str = "full") -> str:
    """Format datetime for display purposes."""
    if dt is None:
        return "Unknown"

    if format_type == "date":
        return dt.strftime("%Y-%m-%d")
    elif format_type == "time":
        milliseconds_str = f".{dt.microsecond // 1000:03d}"
        return dt.strftime("%H:%M:%S") + milliseconds_str
    else:  # full format
        milliseconds_str = f".{dt.microsecond // 1000:03d}"
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S") + milliseconds_str
        timezone_offset = dt.strftime("%z")
        if timezone_offset:
            timezone_formatted = f"{timezone_offset[:3]}:{timezone_offset[3:5]}"
        else:
            timezone_formatted = "+00:00"
        return f"{date_str} {time_str}{timezone_formatted}"


def get_log_timestamp_for_sorting(log_entry: dict) -> float:
    """Extract timestamp from log entry for sorting purposes."""
    parsed_time = parse_timestamp(log_entry.get("timestamp", ""))
    if parsed_time is None:
        return 0.0
    return parsed_time.timestamp()


@st.cache_data
def load_logs_from_file(logfile_path: pathlib.Path) -> pd.DataFrame:
    """Load logs from JSONL file into pandas DataFrame."""
    _logger.debug(f"Loading logs from: {logfile_path}")

    if not logfile_path.exists():
        _logger.error(f"Log file not found: {logfile_path}")
        st.error(f"Log file not found: {logfile_path}")
        return pd.DataFrame()

    log_entries = []
    try:
        with logfile_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        log_entry = json.loads(line)
                        # Parse timestamp for sorting
                        log_entry["parsed_timestamp"] = parse_timestamp(
                            log_entry.get("timestamp", "")
                        )
                        # Add line number for reference
                        log_entry["line_number"] = line_number
                        log_entries.append(log_entry)
                    except json.JSONDecodeError as error:
                        _logger.warning(
                            f"Skipping invalid JSON on line {line_number}: {error}"
                        )
                        st.warning(
                            f"Skipping invalid JSON on line {line_number}: {error}"
                        )
                        continue
    except Exception as error:
        _logger.error(f"Error reading log file: {error}")
        st.error(f"Error reading log file: {error}")
        return pd.DataFrame()

    if not log_entries:
        _logger.warning("No valid log entries found in the file")
        st.warning("No valid log entries found in the file")
        return pd.DataFrame()

    # Create DataFrame and sort by timestamp (newest first)
    dataframe = pd.DataFrame(log_entries)
    dataframe = dataframe.sort_values(
        "parsed_timestamp", ascending=False, na_position="last"
    )
    dataframe = dataframe.reset_index(drop=True)

    _logger.info(f"Loaded {len(dataframe)} log entries from {logfile_path}")
    return dataframe


def apply_log_filters(
    dataframe: pd.DataFrame, message_filter: str, level_filter: str
) -> pd.DataFrame:
    """Apply filtering to the DataFrame based on message and level."""
    if dataframe.empty:
        return dataframe

    filtered_dataframe = dataframe.copy()

    # Apply message filter
    if message_filter:
        message_mask = filtered_dataframe["message"].str.contains(
            message_filter, case=False, na=False
        )
        filtered_dataframe = filtered_dataframe[message_mask]

    # Apply level filter
    if level_filter and level_filter != "ALL":
        level_mask = filtered_dataframe["level"].str.upper() == level_filter.upper()
        filtered_dataframe = filtered_dataframe[level_mask]

    return filtered_dataframe


def calculate_log_statistics(dataframe: pd.DataFrame) -> dict[str, int]:
    """Calculate statistics for log levels."""
    if dataframe.empty:
        return {}

    level_counts = defaultdict(int)
    for level in dataframe["level"]:
        level_counts[level.upper()] += 1

    return dict(level_counts)


def render_log_level_with_styling(level: str) -> str:
    """Render log level with appropriate CSS styling."""
    level_upper = level.upper()
    css_class = f"log-level-{level_upper.lower()}"
    return f'<span class="{css_class}">{level_upper}</span>'


def render_log_message_with_highlighting(message: str, search_term: str = "") -> str:
    """Render log message with search term highlighting."""
    if search_term and search_term in message.lower():
        # Apply highlighting to search term
        highlighted_message = message.replace(
            search_term,
            f'<mark style="background-color: #fbbf24; color: #000;">{search_term}</mark>',
        )
        return f'<span class="log-message">{highlighted_message}</span>'
    return f'<span class="log-message">{message}</span>'


def render_detailed_log_view(log_entry: dict) -> None:
    """Render detailed view of a log entry with all information."""
    column1, column2 = st.columns(2)

    with column1:
        st.markdown("**Primary Information:**")
        st.text(f"Level: {log_entry.get('level', 'Unknown')}")
        st.text(f"Timestamp: {log_entry.get('timestamp', 'Unknown')}")
        st.text(f"Name: {log_entry.get('name', 'Unknown')}")
        st.text(f"Function: {log_entry.get('function', 'Unknown')}")
        st.text(f"Filename: {log_entry.get('filename', 'Unknown')}")
        st.text(f"Line: {log_entry.get('lineno', 'Unknown')}")

    with column2:
        st.markdown("**System Information:**")
        system_information = log_entry.get("system", {})
        st.text(f"Process ID: {log_entry.get('process', 'Unknown')}")
        st.text(f"Thread ID: {log_entry.get('thread', 'Unknown')}")
        st.text(f"Hostname: {system_information.get('hostname', 'Unknown')}")
        st.text(f"Platform: {system_information.get('platform', 'Unknown')}")
        st.text(f"Python: {system_information.get('python_version', 'Unknown')}")

    st.markdown("**Message:**")
    st.code(log_entry.get("message", ""), language=None)

    # Show exception if present
    if "exception" in log_entry and log_entry["exception"]:
        st.markdown("**Exception:**")
        st.code(log_entry["exception"], language=None)

    # Show extra fields if present
    extra_fields = log_entry.get("extra", {})
    if extra_fields and any(value is not None for value in extra_fields.values()):
        st.markdown("**Extra Fields:**")
        st.json(extra_fields)

    # Raw JSON view
    with st.expander("Raw JSON"):
        st.json(log_entry)


def determine_log_file_to_use(provided_path: pathlib.Path | None) -> pathlib.Path:
    """Determine which log file to use based on provided path or auto-discovery."""
    _logger.debug(f"Determining log file to use, provided path: {provided_path}")

    if provided_path and provided_path.exists():
        _logger.info(f"Using provided log file: {provided_path}")
        return provided_path

    # Try to get the default log file location
    try:
        default_log_file = _get_log_file()
        if default_log_file.exists():
            _logger.info(f"Using default log file: {default_log_file}")
            return default_log_file
    except Exception as error:
        _logger.warning(f"Could not get default log file: {error}")

    # Try to find the latest log file in the logs directory
    try:
        default_log_file = _get_log_file()
        log_directory = default_log_file.parent
        latest_log = find_latest_log_file(log_directory)
        if latest_log:
            _logger.info(f"Using latest log file: {latest_log}")
            return latest_log
    except Exception as error:
        _logger.warning(f"Could not find latest log file: {error}")

    # Fallback to the default path (even if it doesn't exist)
    fallback_log = _get_log_file()
    _logger.info(f"Using fallback log file: {fallback_log}")
    return fallback_log


def run_streamlit_app():
    """Main Streamlit application function."""
    _logger.info("Starting log viewer Streamlit app")

    # Apply custom CSS styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Page configuration
    st.set_page_config(
        page_title="Astro Log Viewer",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load custom CSS if available
    try:
        css_path = get_module_dir(__file__) / "streamlit.css"
        if css_path.exists():
            with open(css_path) as file:
                css = file.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        else:
            _logger.debug("No custom CSS file found, using default styling")
    except Exception as error:
        _logger.warning(f"Could not load custom CSS: {error}")

    # Header
    st.title("📋 Astro Log Viewer")

    # Initialize session state for log file path
    if "logfile_path" not in st.session_state:
        # Check URL parameters first
        query_params = st.query_params
        if "file" in query_params:
            st.session_state.logfile_path = pathlib.Path(query_params["file"])
            _logger.debug(
                f"Using log file from URL parameter: {st.session_state.logfile_path}"
            )
        else:
            st.session_state.logfile_path = determine_log_file_to_use(None)

    current_logfile_path = st.session_state.logfile_path

    # Sidebar for controls and filters
    with st.sidebar:
        st.header("🔧 Controls")

        # File selector section
        st.subheader("Log File Selection")

        try:
            log_directory = current_logfile_path.parent
            available_log_files = get_available_log_files(log_directory)

            if available_log_files:
                # Create a selectbox with file names and modification times
                file_options = []
                file_paths = []

                for log_file in available_log_files:
                    try:
                        modification_time = datetime.fromtimestamp(
                            log_file.stat().st_mtime
                        )
                        time_str = modification_time.strftime("%Y-%m-%d %H:%M:%S")
                        file_display = f"{log_file.name} ({time_str})"
                        file_options.append(file_display)
                        file_paths.append(log_file)
                    except Exception:
                        file_options.append(log_file.name)
                        file_paths.append(log_file)

                # Find current selection index
                try:
                    current_index = file_paths.index(current_logfile_path)
                except ValueError:
                    current_index = 0

                selected_index = st.selectbox(
                    "Choose log file:",
                    range(len(file_options)),
                    format_func=lambda index: file_options[index],
                    index=current_index,
                    key="file_selector",
                )

                # Update session state if selection changed
                if file_paths[selected_index] != current_logfile_path:
                    st.session_state.logfile_path = file_paths[selected_index]
                    st.query_params["file"] = str(file_paths[selected_index])
                    _logger.info(f"Switched to log file: {file_paths[selected_index]}")
                    st.rerun()

            else:
                st.warning("No log files found in the directory")
                st.text(f"Directory: {log_directory}")

        except Exception as error:
            _logger.error(f"Error accessing log directory: {error}")
            st.error(f"Error accessing log directory: {error}")

        # Current file info
        st.text(f"Current: {current_logfile_path.name}")

        # Refresh button
        if st.button("🔄 Reload Logs"):
            _logger.info("Reloading logs requested by user")
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Filters section
        st.subheader("🔍 Filters")

        message_filter = st.text_input(
            "Filter by message:",
            placeholder="Enter text to search...",
            key="message_filter",
        )

        level_filter = st.selectbox(
            "Filter by level:",
            ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            key="level_filter",
        )

        # Clear filters button
        if st.button("🗑️ Clear Filters"):
            _logger.info("Clearing filters requested by user")
            st.session_state.message_filter = ""
            st.session_state.level_filter = "ALL"
            st.rerun()

    # Load logs
    with st.spinner("Loading logs..."):
        dataframe = load_logs_from_file(current_logfile_path)

    if dataframe.empty:
        st.warning("No logs to display")
        return

    # Apply filters
    filtered_dataframe = apply_log_filters(dataframe, message_filter, level_filter)

    # Calculate statistics
    statistics = calculate_log_statistics(filtered_dataframe)
    total_log_count = len(dataframe)
    displayed_log_count = len(filtered_dataframe)

    _logger.debug(f"Displaying {displayed_log_count} of {total_log_count} logs")

    # Display statistics in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("📊 Statistics")

        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.metric("Total", total_log_count)
        with stats_col2:
            st.metric("Showing", displayed_log_count)

        if statistics:
            st.markdown("**By Level:**")
            for level, count in sorted(statistics.items()):
                color = LEVEL_COLORS.get(level, "#6b7280")
                st.markdown(
                    f'<div style="color: {color}; font-weight: bold;">{level}: {count}</div>',
                    unsafe_allow_html=True,
                )

    # Main content area
    if filtered_dataframe.empty:
        st.info("No logs match the current filters")
        return

    # Display logs section
    st.subheader(f"Log Entries ({displayed_log_count} entries)")

    # Pagination controls
    logs_per_page = st.selectbox("Logs per page:", [25, 50, 100, 200], index=1)

    total_pages = (len(filtered_dataframe) - 1) // logs_per_page + 1

    if total_pages > 1:
        page_number = st.selectbox(
            f"Page (1-{total_pages}):", range(1, total_pages + 1)
        )
        start_index = (page_number - 1) * logs_per_page
        end_index = start_index + logs_per_page
        page_dataframe = filtered_dataframe.iloc[start_index:end_index]
    else:
        page_dataframe = filtered_dataframe

    # Display log entries
    for index, (_, log_entry) in enumerate(page_dataframe.iterrows()):
        with st.container():
            # Create header for the log entry
            timestamp_formatted = format_datetime_for_display(
                log_entry.get("parsed_timestamp")
            )
            level = log_entry.get("level", "UNKNOWN")
            name = log_entry.get("name", "Unknown")
            message = log_entry.get("message", "")

            # Display log entry header in columns
            header_col1, header_col2, header_col3, header_col4 = st.columns(
                [2, 1, 2, 4]
            )

            with header_col1:
                st.markdown(
                    f'<span class="log-timestamp">{timestamp_formatted}</span>',
                    unsafe_allow_html=True,
                )
            with header_col2:
                st.markdown(
                    render_log_level_with_styling(level), unsafe_allow_html=True
                )
            with header_col3:
                st.markdown(
                    f'<span class="log-name">{name}</span>', unsafe_allow_html=True
                )
            with header_col4:
                st.markdown(
                    render_log_message_with_highlighting(message, message_filter),
                    unsafe_allow_html=True,
                )

            # Expandable details section
            with st.expander(
                f"Details for entry #{log_entry.get('line_number', index + 1)}"
            ):
                render_detailed_log_view(log_entry)

            # Add visual separator
            st.divider()


@click.command()
@click.argument("logfile_path", type=click.Path(path_type=pathlib.Path), required=False)
@click.option(
    "--port",
    default=8502,
    show_default=True,
    type=int,
    help="Port to run the Streamlit server on (1024-65535)",
)
@click.option(
    "--host",
    default="localhost",
    show_default=True,
    help="Host address for the Streamlit server (use '0.0.0.0' for all interfaces)",
)
@click.option(
    "--no-browser/--browser",
    default=False,
    show_default=True,
    help="If set, do not open the browser automatically on startup",
)
@click.option(
    "--debug-mode/--no-debug-mode",
    default=False,
    show_default=True,
    help="Enable Streamlit debug mode (shows extra logs and tracebacks)",
)
@click.option(
    "--theme",
    default="dark",
    show_default=True,
    type=click.Choice(["light", "dark", "system"], case_sensitive=False),
    help="Streamlit theme: 'light', 'dark' or 'system'",
)
def view_logs(
    logfile_path: pathlib.Path | None,
    port: int,
    host: str,
    no_browser: bool,
    debug_mode: bool,
    theme: str,
):
    """Launch the Streamlit log viewer for the specified log file.

    If no logfile is provided, the viewer will automatically find and use
    the most recent log file in the logs directory.
    """
    _logger.info("Starting Astro Log Viewer")

    if runtime.exists():
        _logger.debug("Running in Streamlit runtime")
        # Store the logfile path for the app
        if logfile_path:
            st.session_state.logfile_path = logfile_path
        run_streamlit_app()
    else:
        # Determine which log file to use
        target_logfile = determine_log_file_to_use(logfile_path)

        if not target_logfile.exists():
            _logger.warning(f"Log file not found: {target_logfile}")
            click.echo(f"Warning: Log file not found: {target_logfile}")
            click.echo("The viewer will still start, but no logs will be displayed.")

        _logger.info(f"Starting log viewer for: {target_logfile}")
        click.echo(f"Starting Astro Log Viewer for: {target_logfile}")
        click.echo("The viewer will open in your browser...")

        # Build Streamlit command arguments
        sys.argv = [
            "streamlit",
            "run",
            __file__,
            "--server.port",
            str(port),
            "--server.address",
            host,
            "--server.headless",
            "true" if no_browser else "false",
            "--server.runOnSave",
            "true",
            "--theme.base",
            theme,
            "--",
            str(target_logfile),
        ]

        if no_browser:
            sys.argv += ["--browser.gatherUsageStats", "false"]
        if debug_mode:
            sys.argv += ["--global.developmentMode", "true"]

        _logger.debug(f"Starting Streamlit with args: {sys.argv}")
        sys.exit(stcli.main())


if __name__ == "__main__":
    # Check if we're running via streamlit (sys.argv will contain streamlit-specific args)
    if len(sys.argv) > 1 and sys.argv[1] != "--help":
        # Check if there's a logfile argument passed after --
        dash_dash_index = None
        try:
            dash_dash_index = sys.argv.index("--")
            if dash_dash_index + 1 < len(sys.argv):
                provided_logfile = pathlib.Path(sys.argv[dash_dash_index + 1])
                st.session_state.logfile_path = provided_logfile
        except (ValueError, IndexError):
            # No -- argument found or no logfile after it
            pass

        # Run the main app
        run_streamlit_app()
    else:
        # Running via click command
        view_logs()
