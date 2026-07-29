"""Command-line interface.

A thin adapter over the service layer: every command resolves arguments, calls a service
and renders the result. No cataloguing logic lives here, which is what keeps the CLI and
the HTTP API from drifting apart.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ai_asset_manager import __version__
from ai_asset_manager.backend.database.engine import configure_engine, get_engine, session_scope
from ai_asset_manager.backend.database.schema import init_database
from ai_asset_manager.backend.inventory import (
    DEFAULT_LEVELS,
    GROUP_BY_FIELDS,
    SORT_FIELDS,
    TREE_LEVELS,
    InventoryEngine,
    available_formats,
    build_tree,
    export_report,
    known_aliases,
    resolve_alias,
    suggest_filename,
)
from ai_asset_manager.backend.inventory.render import (
    render_dataset_table,
    render_details,
    render_distribution,
    render_fixes,
    render_group_heading,
    render_health,
    render_storage_breakdown,
    render_summary,
    render_table,
    render_tree,
)
from ai_asset_manager.backend.models import Asset, ScanRun
from ai_asset_manager.backend.scanner.progress import ScanContext, ScanPhase, ScanProgress
from ai_asset_manager.backend.services.asset_service import AssetFilter, AssetService
from ai_asset_manager.backend.services.scan_service import ScanService
from ai_asset_manager.backend.utils.humanize import (
    format_bytes,
    format_count,
    format_duration,
    format_relative_time,
    parse_size,
)
from ai_asset_manager.backend.utils.paths import shorten_path
from ai_asset_manager.config import get_settings
from ai_asset_manager.logging_conf import configure_logging

#: Help panels, so that twelve commands read as three short groups rather than one long
#: alphabetical list. Which group a command is in is the first thing that tells a new user
#: what to run.
PANEL_START = "Getting started"
PANEL_LIBRARY = "Your library"
PANEL_ABOUT = "About"

#: The inventory command has fifteen options. Split into three panels they read as a
#: menu; in one flat list they read as a wall, and the useful ones get lost.
OPTS_FILTER = "Choosing what to show"
OPTS_LAYOUT = "Choosing how to show it"
OPTS_OUTPUT = "Taking it elsewhere"

app = typer.Typer(
    name="aam",
    help=(
        "[bold]AI Asset Manager[/bold] - find out what AI models and datasets are on this "
        "machine, what they are for, and which of them are broken."
    ),
    # Typer renders the epilog as flowing text and collapses single newlines, so this has
    # to read as a sentence rather than a table. The table lives in `aam guide`.
    epilog=(
        "Start with [cyan]aam scan --auto[/cyan], then [cyan]aam inventory[/cyan]. "
        "[cyan]aam guide[/cyan] has worked examples. "
        "Everything here is read-only: nothing is ever moved, renamed or deleted."
    ),
    no_args_is_help=False,
    add_completion=True,
    rich_markup_mode="rich",
)
roots_app = typer.Typer(
    help="Manage the folders that get scanned.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(roots_app, name="roots", rich_help_panel=PANEL_START)

# Windows consoles default to a legacy code page, and model names routinely contain
# characters it cannot encode. Rich handles the encoding; forcing it here means a name
# with CJK or emoji in it prints as replacement characters instead of aborting the
# command with a UnicodeEncodeError.
console = Console(soft_wrap=False)
error_console = Console(stderr=True, style="bold red")


#: Commands that answer without touching the catalogue. They must not create one:
#: a freshly downloaded binary being asked its version should not leave an empty
#: database and a data directory behind on a machine it may never run on again.
COMMANDS_WITHOUT_DATABASE = frozenset({"version"})


def _bootstrap(
    *, verbose: bool = False, database: Path | None = None, schema: bool = True
) -> None:
    """Configure logging and, unless told otherwise, ensure the database exists."""
    settings = get_settings()
    configure_logging(
        "DEBUG" if verbose else settings.log_level,
        log_file=settings.log_file,
    )
    if database is not None:
        engine = configure_engine(f"sqlite+pysqlite:///{database.as_posix()}")
    elif schema:
        settings.ensure_data_dir()
        engine = get_engine()
    else:
        return

    if schema:
        init_database(engine)


def _version_callback(value: bool) -> None:
    """Print the version and exit, for the conventional ``--version`` flag."""
    if value:
        console.print(f"AI Asset Manager {__version__}")
        raise typer.Exit


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
    database: Annotated[
        Path | None, typer.Option("--database", help="Use a specific SQLite file.")
    ] = None,
    # Consumed entirely by its eager callback, which prints and exits before this body
    # runs; the parameter exists so that Typer registers the flag at all.
    show_version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True,
                     help="Show the version and exit."),
    ] = False,
) -> None:
    """Set up logging and the database before any command runs."""
    bare = ctx.invoked_subcommand is None
    _bootstrap(
        verbose=verbose,
        database=database,
        schema=not bare and ctx.invoked_subcommand not in COMMANDS_WITHOUT_DATABASE,
    )
    if bare:
        _welcome(database)
        raise typer.Exit


def _welcome(database: Path | None) -> None:
    """Greet a bare ``aam`` with the next thing worth doing.

    Printing the help text here would be the conventional choice and the less useful one.
    Someone typing ``aam`` with nothing after it does not want a list of twelve commands;
    they want to know whether this thing has anything to tell them yet, and if not, how to
    make it. So the answer depends on whether a catalogue exists.
    """
    settings = get_settings()
    path = database or settings.db_path

    total = _catalogue_size(path)

    if total is None:
        _print_first_run()
        return

    console.print(f"[bold]AI Asset Manager[/bold] [dim]{__version__}[/dim]")
    console.print(f"{total:,} asset(s) catalogued.\n")
    console.print("  [cyan]aam inventory[/cyan]          what you have, and what it is for")
    console.print("  [cyan]aam inventory --tree[/cyan]   the shape of the whole library")
    console.print("  [cyan]aam inventory missing[/cyan]  what needs attention")
    console.print("  [cyan]aam where <name>[/cyan]       where did I put it?")
    console.print("\n[dim]aam --help for everything, aam guide for worked examples.[/dim]")


def _catalogue_size(path: Path) -> int | None:
    """Return how many assets are catalogued, or ``None`` if there is no catalogue yet.

    Deliberately opens the file read-only and by hand rather than going through the ORM:
    asking "is there anything here?" must not be the thing that creates an empty database
    on a machine the tool has only just been copied to.
    """
    if not path.exists():
        return None

    import sqlite3

    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT count(*) FROM assets WHERE is_missing = 0"
            ).fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else 0


def _print_first_run() -> None:
    """Explain what this is and offer concrete folders to scan."""
    from ai_asset_manager.backend.scanner.locations import likely_asset_folders

    console.print(f"[bold]AI Asset Manager[/bold] [dim]{__version__}[/dim]")
    console.print(
        "Nothing catalogued yet. Point it at a folder and it works out what is in there "
        "-\nmodels, datasets, adapters, which are duplicates and which are broken.\n"
    )

    found = likely_asset_folders()
    if found:
        table = Table(box=None, pad_edge=False, show_header=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="dim")
        for location in found[:8]:
            table.add_row(str(location.path), location.label)

        console.print("Found these on this machine:")
        console.print(table)
        console.print("\n  [cyan]aam scan --auto[/cyan]   [dim]catalogue all of them[/dim]")
    else:
        console.print("  [cyan]aam scan D:\\Models[/cyan]   [dim]catalogue a folder[/dim]")

    console.print(
        "\n[dim]Scanning only reads. Nothing is moved, renamed or deleted, ever.[/dim]"
    )


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


@app.command(rich_help_panel=PANEL_START)
def scan(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Folders to scan. Defaults to the registered roots."),
    ] = None,
    auto: Annotated[
        bool,
        typer.Option("--auto", help="Scan the model and dataset caches found on this machine."),
    ] = False,
    full: Annotated[
        bool, typer.Option("--full", help="Re-parse every asset, ignoring fingerprints.")
    ] = False,
    add: Annotated[
        bool, typer.Option("--add", help="Also register these paths as permanent scan roots.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress the progress bar.")
    ] = False,
) -> None:
    r"""Scan folders and update the catalogue.

    Rescans are incremental: assets whose files are unchanged are skipped entirely.

    [bold]Examples[/bold]

      [cyan]aam scan --auto[/cyan]              every cache this tool can find
      [cyan]aam scan D:\\Models --add[/cyan]     a folder, remembered for next time
      [cyan]aam scan[/cyan]                     everything remembered
    """
    targets = [str(path) for path in paths] if paths else None

    if auto:
        from ai_asset_manager.backend.scanner.locations import likely_asset_folders

        discovered = likely_asset_folders()
        if not discovered:
            error_console.print(
                "Found no known model or dataset folders on this machine.\n"
                "Pass one directly: aam scan D:\\Models"
            )
            raise typer.Exit(code=1)

        console.print(f"Found {len(discovered)} location(s):")
        for location in discovered:
            console.print(f"  [cyan]{location.path}[/cyan]  [dim]{location.label}[/dim]")
        console.print()
        targets = (targets or []) + [str(location.path) for location in discovered]

    with session_scope() as session:
        service = ScanService(session)

        if add and targets:
            for target in targets:
                service.add_root(target)
            session.commit()

        if not targets and not service.list_roots(enabled_only=True):
            error_console.print(
                "No scan roots configured.\n"
                "Try 'aam scan --auto' to catalogue the caches on this machine, pass a "
                "folder directly, or add one with 'aam roots add <path>'."
            )
            raise typer.Exit(code=1)

        cancel_event = threading.Event()
        if quiet:
            context = ScanContext(cancel_event=cancel_event)
            run = service.scan(targets, context=context, incremental=not full)
        else:
            run = _scan_with_progress(service, targets, cancel_event, incremental=not full)

    _print_scan_summary(run)


def _scan_with_progress(
    service: ScanService,
    targets: list[str] | None,
    cancel_event: threading.Event,
    *,
    incremental: bool,
) -> ScanRun:
    """Run a scan while rendering a live progress bar."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("{task.fields[detail]}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Starting", total=None, detail="")

        def on_progress(snapshot: ScanProgress) -> None:
            """Mirror scan state into the progress bar."""
            if snapshot.phase in (ScanPhase.WALKING, ScanPhase.STARTING):
                detail = (
                    f"{snapshot.files_seen:,} files · "
                    f"{format_bytes(snapshot.bytes_seen)} · "
                    f"{shorten_path(snapshot.current_path, 44)}"
                )
                progress.update(task, description="Walking", total=None, detail=detail)
            else:
                detail = f"{snapshot.assets_found} assets · {snapshot.errors} errors"
                progress.update(
                    task,
                    description=snapshot.phase.value.title(),
                    total=snapshot.total or None,
                    completed=snapshot.completed,
                    detail=detail,
                )

        context = ScanContext(on_progress=on_progress, cancel_event=cancel_event)
        try:
            return service.scan(targets, context=context, incremental=incremental)
        except KeyboardInterrupt:
            # Signal the workers and let the service finish writing a CANCELLED run,
            # rather than leaving the database mid-transaction.
            cancel_event.set()
            console.print("[yellow]Cancelling…[/yellow]")
            raise


def _print_scan_summary(run: ScanRun) -> None:
    """Render the outcome of a scan run."""
    style = {"completed": "green", "cancelled": "yellow"}.get(run.status, "red")
    lines = [
        f"[bold]{run.assets_found}[/bold] assets  "
        f"([green]{run.assets_created} new[/green], "
        f"{run.assets_updated} updated, "
        f"[dim]{run.assets_unchanged} unchanged[/dim])",
        f"{run.files_seen:,} files · {format_bytes(run.bytes_seen)} · "
        f"{format_duration(run.duration_seconds or 0)}",
    ]
    if run.assets_missing:
        lines.append(f"[yellow]{run.assets_missing} asset(s) no longer on disk[/yellow]")
    if run.error_count:
        lines.append(f"[red]{run.error_count} error(s)[/red]")
    if run.message:
        lines.append(f"[dim]{run.message}[/dim]")

    console.print(
        Panel("\n".join(lines), title=f"Scan {run.status}", border_style=style, expand=False)
    )


# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------


@roots_app.command("add")
def roots_add(
    paths: Annotated[list[Path], typer.Argument(help="Folders to register.")],
    label: Annotated[str | None, typer.Option(help="Friendly name for the root.")] = None,
) -> None:
    """Register folders to be scanned."""
    with session_scope() as session:
        service = ScanService(session)
        for path in paths:
            if not path.exists():
                error_console.print(f"Does not exist: {path}")
                continue
            root = service.add_root(str(path), label=label)
            console.print(f"[green]Added[/green] {root.path}")


@roots_app.command("list")
def roots_list() -> None:
    """List the registered scan roots."""
    with session_scope() as session:
        roots = ScanService(session).list_roots()
        if not roots:
            console.print("[dim]No scan roots registered.[/dim]")
            return

        table = Table(box=None, pad_edge=False)
        table.add_column("Path", style="cyan", overflow="fold")
        table.add_column("Label")
        table.add_column("Enabled", justify="center")
        table.add_column("Assets", justify="right")
        table.add_column("Last scanned", style="dim")

        for root in roots:
            table.add_row(
                root.path,
                root.label or "",
                "yes" if root.enabled else "no",
                str(root.last_asset_count),
                format_relative_time(root.last_scanned) if root.last_scanned else "never",
            )
        console.print(table)


@roots_app.command("remove")
def roots_remove(
    path: Annotated[Path, typer.Argument(help="Folder to unregister.")],
) -> None:
    """Unregister a scan root. Catalogued assets are kept."""
    with session_scope() as session:
        if ScanService(session).remove_root(str(path)):
            console.print(f"[green]Removed[/green] {path}")
        else:
            error_console.print(f"Not a registered root: {path}")
            raise typer.Exit(code=1)


# --------------------------------------------------------------------------
# Browsing
# --------------------------------------------------------------------------


@app.command("list", rich_help_panel=PANEL_LIBRARY)
def list_assets(
    kind: Annotated[str | None, typer.Option(help="model, dataset, adapter, checkpoint.")] = None,
    model_type: Annotated[str | None, typer.Option(help="llm, vision_language, embedding…")] = None,
    framework: Annotated[str | None, typer.Option(help="transformers, gguf, ultralytics…")] = None,
    drive: Annotated[str | None, typer.Option(help="Restrict to one drive, e.g. 'F:'.")] = None,
    tag: Annotated[str | None, typer.Option(help="Restrict to assets carrying a tag.")] = None,
    min_size: Annotated[str | None, typer.Option(help="Minimum size, e.g. '1GB'.")] = None,
    search: Annotated[str | None, typer.Option("--search", "-s", help="Free-text match.")] = None,
    sort: Annotated[str, typer.Option(help="size, name, modified, created, files.")] = "size",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Rows to show.")] = 30,
    paths: Annotated[
        bool, typer.Option("--paths", help="Show full paths instead of details.")
    ] = False,
) -> None:
    """List catalogued assets."""
    filters = AssetFilter(
        text=search,
        kinds=[kind] if kind else [],
        model_types=[model_type] if model_type else [],
        frameworks=[framework] if framework else [],
        drives=[drive.upper()] if drive else [],
        tags=[tag] if tag else [],
        min_size=parse_size(min_size) if min_size else None,
    )

    with session_scope() as session:
        page = AssetService(session).list(filters, limit=limit, sort=sort)

        if not page.items:
            console.print("[dim]No matching assets. Run 'aam scan' first?[/dim]")
            return

        console.print(_asset_table(page.items, show_paths=paths))
        if page.has_more:
            console.print(
                f"[dim]Showing {len(page.items)} of {page.total}. "
                f"Use --limit to see more.[/dim]"
            )


def _asset_table(assets: Sequence[Asset], *, show_paths: bool) -> Table:
    """Render assets as a table."""
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Name", style="cyan", max_width=42, overflow="ellipsis")
    table.add_column("Kind", max_width=10)
    table.add_column("Size", justify="right")

    if show_paths:
        table.add_column("Path", overflow="fold", style="dim")
    else:
        table.add_column("Type")
        table.add_column("Params", justify="right")
        table.add_column("Format")
        table.add_column("Modified", style="dim")

    for asset in assets:
        row = [
            str(asset.id),
            asset.display_name or asset.name,
            asset.kind,
            format_bytes(asset.size_bytes),
        ]
        if show_paths:
            row.append(asset.root_path)
        else:
            details = asset.model_details
            dataset = asset.dataset_details
            if details is not None:
                type_label = details.model_type
                params = (
                    format_count(details.param_count)
                    + ("" if details.param_count_is_exact else "~")
                    if details.param_count
                    else ""
                )
            elif dataset is not None:
                type_label = dataset.dataset_format
                params = f"{dataset.num_images:,} img" if dataset.num_images else ""
            else:
                type_label, params = "", ""

            row.extend(
                [
                    type_label,
                    params,
                    asset.format,
                    format_relative_time(asset.modified_at) if asset.modified_at else "",
                ]
            )
        table.add_row(*row)
    return table


@app.command(rich_help_panel=PANEL_LIBRARY)
def show(
    asset_id: Annotated[int, typer.Argument(help="Asset id, as shown by 'aam list'.")],
    files: Annotated[bool, typer.Option("--files", help="List the asset's files.")] = False,
) -> None:
    """Show everything known about one asset."""
    with session_scope() as session:
        asset = AssetService(session).get(asset_id)
        if asset is None:
            error_console.print(f"No asset with id {asset_id}")
            raise typer.Exit(code=1)

        console.print(_detail_panel(asset))

        if files:
            table = Table(box=None, pad_edge=False, header_style="bold")
            table.add_column("File", overflow="fold")
            table.add_column("Size", justify="right")
            for record in sorted(asset.files, key=lambda item: -item.size_bytes)[:200]:
                table.add_row(record.relpath, format_bytes(record.size_bytes))
            console.print(table)

        if asset.health_findings:
            console.print("\n[bold]Health[/bold]")
            for finding in asset.health_findings:
                colour = {"error": "red", "warning": "yellow"}.get(finding.severity, "dim")
                console.print(f"  [{colour}]{finding.severity}[/{colour}] {finding.message}")


def _detail_panel(asset: Asset) -> Panel:
    """Render an asset's metadata as a panel."""
    lines = [
        f"[bold cyan]{asset.display_name or asset.name}[/bold cyan]",
        f"[dim]{asset.root_path}[/dim]",
        "",
        f"{'Kind':<14}{asset.kind}" + (f" / {asset.subkind}" if asset.subkind else ""),
        f"{'Size':<14}{format_bytes(asset.size_bytes)} across {asset.file_count} file(s)",
    ]
    if asset.physical_size_bytes and asset.physical_size_bytes != asset.size_bytes:
        lines.append(f"{'On disk':<14}{format_bytes(asset.physical_size_bytes)} (shared storage)")

    lines.extend(
        [
            f"{'Format':<14}{asset.format}",
            f"{'Framework':<14}{asset.framework}",
            f"{'Drive':<14}{asset.drive or 'unknown'}",
            f"{'Detected by':<14}{asset.detector} ({asset.detection_confidence:.0%} confidence)",
        ]
    )

    details = asset.model_details
    if details is not None:
        lines.append("")
        for label, value in (
            ("Type", details.model_type),
            ("Architecture", details.architecture),
            (
                "Parameters",
                format_count(details.param_count)
                + ("" if details.param_count_is_exact else " (estimated)")
                if details.param_count
                else None,
            ),
            ("Quantization", details.quantization),
            ("Precision", details.precision),
            ("Context", f"{details.context_length:,}" if details.context_length else None),
            ("Layers", details.num_layers),
            ("Repository", details.repo_id),
            ("Author", details.author),
            ("License", details.license),
            ("Base model", details.base_model),
        ):
            if value:
                lines.append(f"{label:<14}{value}")
        if details.description:
            lines.extend(["", f"[dim]{details.description[:400]}[/dim]"])

    dataset = asset.dataset_details
    if dataset is not None:
        lines.append("")
        for label, value in (
            ("Format", dataset.dataset_format),
            ("Task", dataset.task),
            ("Images", f"{dataset.num_images:,}" if dataset.num_images else None),
            ("Videos", f"{dataset.num_videos:,}" if dataset.num_videos else None),
            ("Annotations", f"{dataset.num_annotations:,}" if dataset.num_annotations else None),
            ("Classes", dataset.num_classes),
            ("Splits", ", ".join(f"{k}={v:,}" for k, v in dataset.splits.items()) or None),
            ("Modalities", ", ".join(dataset.modalities) or None),
        ):
            if value:
                lines.append(f"{label:<14}{value}")
        if dataset.class_names:
            preview = ", ".join(dataset.class_names[:15])
            extra_count = len(dataset.class_names) - 15
            suffix = f" … (+{extra_count})" if extra_count > 0 else ""
            lines.extend(["", f"[dim]Classes: {preview}{suffix}[/dim]"])

    if asset.tags:
        lines.extend(["", "Tags".ljust(14) + ", ".join(tag.name for tag in asset.tags)])

    return Panel("\n".join(lines), border_style="cyan", expand=False)


@app.command(rich_help_panel=PANEL_LIBRARY)
def inventory(
    category: Annotated[
        str,
        typer.Argument(
            help="What to list: all, models, datasets, llm, ocr, vision, speech, "
            "embeddings, detection, segmentation, tracking, diffusion, adapters, "
            "experiments, any *-datasets variant, or a domain such as medical. "
            "Two special views: 'health' and 'missing'.",
        ),
    ] = "all",
    group_by: Annotated[
        str | None,
        typer.Option("--group-by", "-g", help=f"Group by: {', '.join(GROUP_BY_FIELDS)}.",
                     rich_help_panel=OPTS_LAYOUT),
    ] = None,
    sort: Annotated[
        str, typer.Option("--sort", help=f"Sort by: {', '.join(SORT_FIELDS)}.",
                     rich_help_panel=OPTS_LAYOUT)
    ] = "size",
    ascending: Annotated[
        bool, typer.Option("--asc", help="Sort ascending instead of descending.",
                     rich_help_panel=OPTS_LAYOUT)
    ] = False,
    drive: Annotated[
        str | None, typer.Option("--drive", help="Restrict to one drive, e.g. 'F:'.",
                     rich_help_panel=OPTS_FILTER)
    ] = None,
    framework: Annotated[
        str | None, typer.Option("--framework", help="Restrict to one framework.",
                     rich_help_panel=OPTS_FILTER)
    ] = None,
    task: Annotated[
        str | None, typer.Option("--task", help="Restrict to one task, e.g. 'object_detection'.",
                     rich_help_panel=OPTS_FILTER)
    ] = None,
    domain: Annotated[
        str | None, typer.Option("--domain", help="Restrict to one domain, e.g. 'medical'.",
                     rich_help_panel=OPTS_FILTER)
    ] = None,
    details: Annotated[
        bool, typer.Option("--details", "-d", help="Show everything known about each asset.",
                     rich_help_panel=OPTS_LAYOUT)
    ] = False,
    tree: Annotated[
        bool, typer.Option("--tree", "-t", help="Show the library as a tree.",
                     rich_help_panel=OPTS_LAYOUT)
    ] = False,
    tree_by: Annotated[
        str | None,
        typer.Option("--tree-by", help=f"Tree nesting, comma-separated: {', '.join(TREE_LEVELS)}.",
                     rich_help_panel=OPTS_LAYOUT),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Show only the first N assets.",
                     rich_help_panel=OPTS_FILTER)
    ] = None,
    include_missing: Annotated[
        bool, typer.Option("--include-missing", help="Include assets no longer on disk.",
                     rich_help_panel=OPTS_FILTER)
    ] = False,
    export: Annotated[
        str | None,
        typer.Option(
            "--export",
            help=f"Export instead of printing: {', '.join(available_formats())}.",
            rich_help_panel=OPTS_OUTPUT,
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="File to write the export to.",
                     rich_help_panel=OPTS_OUTPUT)
    ] = None,
    storage: Annotated[
        bool, typer.Option("--storage", help="Also show the storage breakdown by drive.",
                     rich_help_panel=OPTS_OUTPUT)
    ] = False,
) -> None:
    """List everything in your local AI library, by category.

    Reads only the catalogue built by 'aam scan' — no folders are walked and no files are
    opened, so it answers instantly however large the library is. Strictly read-only.

    Two views take the place of a category: 'health' scores every asset and lists what is
    wrong with it, and 'missing' shows only the assets that need attention.

    [bold]Examples[/bold]

      [cyan]aam inventory[/cyan]                    everything, by category
      [cyan]aam inventory ocr[/cyan]                which OCR models are installed
      [cyan]aam inventory --tree[/cyan]             the shape of the library
      [cyan]aam inventory --details[/cyan]          everything known about each asset
      [cyan]aam inventory missing[/cyan]            only what needs attention
      [cyan]aam inventory -g family[/cyan]          grouped by model family
      [cyan]aam inventory --export csv -o x.csv[/cyan]  take it to a spreadsheet
    """
    view = category.strip().lower()
    health_view = view in ("health", "missing")

    if health_view:
        selected: Sequence[str] | None = None
    else:
        categories = resolve_alias(category)
        if categories is None:
            _report_unknown_category(category)
            raise typer.Exit(code=2)
        # "all" means no restriction; passing the full category tuple would also filter
        # out anything whose category came from a plugin no longer installed.
        selected = None if view == "all" else categories

    with session_scope() as session:
        report = InventoryEngine(session).build(
            selected,
            drives=[drive] if drive else None,
            frameworks=[framework] if framework else None,
            tasks=[task] if task else None,
            domains=[domain] if domain else None,
            group_by=group_by,
            sort="health" if health_view and sort == "size" else sort,
            descending=not ascending,
            include_missing=include_missing or health_view,
            only_unhealthy=view == "missing",
            limit=limit,
        )

        if export:
            _export_inventory(report, export, output, category)
            return

        if report.is_empty:
            if view == "missing":
                console.print(
                    "[green]Nothing needs attention.[/green] "
                    "[dim]Every catalogued asset is complete.[/dim]"
                )
            else:
                console.print(
                    f"[dim]No assets found for {category!r}. "
                    "Run 'aam scan <folder>' first, or try 'aam inventory all'.[/dim]"
                )
            return

        console.print(render_summary(report))

        if tree or tree_by:
            levels = (
                tuple(level.strip() for level in tree_by.split(",") if level.strip())
                if tree_by
                else DEFAULT_LEVELS
            )
            console.print()
            console.print(render_tree(build_tree(report, levels=levels)))
        elif health_view:
            console.print()
            console.print(render_health(report.items))
            fixes = render_fixes(report.items)
            if fixes is not None:
                console.print()
                console.print(fixes)
        elif report.groups:
            for group in report.groups:
                console.print(render_group_heading(group))
                console.print(_table_for(group.items, details=details))
        else:
            console.print()
            console.print(_table_for(report.items, details=details))

        if storage:
            for axis in ("drive", "framework"):
                breakdown = render_storage_breakdown(report, by=axis)
                if breakdown is not None:
                    console.print()
                    console.print(breakdown)
            for axis in ("task", "family"):
                distribution = render_distribution(report, by=axis)
                if distribution is not None:
                    console.print()
                    console.print(distribution)

        if limit is not None and report.summary.total_assets > len(report.items):
            console.print(
                f"\n[dim]Showing {len(report.items)} of {report.summary.total_assets}. "
                "Raise --limit to see more.[/dim]"
            )


def _report_unknown_category(category: str) -> None:
    """Explain a category name the taxonomy does not recognise.

    Dumping all seventy selectors — which is what the taxonomy now offers, since plugins
    add their own — buries the answer. A close match is nearly always what was meant, so
    that is offered first and the full list is grouped by section behind it.
    """
    import difflib

    from ai_asset_manager.backend.inventory import all_categories, all_sections

    aliases = known_aliases()
    close = difflib.get_close_matches(category.lower().replace("_", "-"), aliases, n=3)

    error_console.print(f"Unknown category {category!r}.")

    if close:
        console.print(f"\nDid you mean: [cyan]{'[/cyan], [cyan]'.join(close)}[/cyan]?")

    console.print("\n[bold]Everything you can ask for[/bold]")
    for section in all_sections():
        names = [category.id for category in all_categories() if category.section == section.id]
        if names:
            console.print(f"  [dim]{section.label:<18}[/dim] {', '.join(names)}")
    console.print(
        "\n[dim]Plus 'all', 'health', 'missing', any domain name, and the shorthands "
        f"{', '.join(a for a in aliases if '-' not in a)[:96]}...[/dim]"
    )


def _table_for(items: Sequence[Any], *, details: bool) -> Table:
    """Choose the layout that fits a set of items.

    Detail mode gets per-asset blocks, since the full field set cannot be read as a row.
    Otherwise datasets get their own columns — they have no architecture or parameter
    count, and a shared layout would show them a row of blanks.
    """
    if details:
        return render_details(items)
    if items and all(item.is_dataset for item in items):
        return render_dataset_table(items)
    return render_table(items, show_details=False)


def _export_inventory(
    report: Any, fmt: str, output: Path | None, category: str
) -> None:
    """Write or print an inventory export."""
    try:
        destination = output or Path(suggest_filename(fmt, prefix=f"inventory-{category}"))
        rendered = export_report(report, fmt, destination)
    except ValueError as exc:
        error_console.print(str(exc))
        raise typer.Exit(code=2) from exc
    except OSError as exc:
        error_console.print(f"Cannot write {output}: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Exported[/green] {report.summary.total_assets} asset(s) "
        f"({format_bytes(report.summary.total_bytes)}) to {destination}"
    )
    if len(rendered) < 400:
        console.print(f"[dim]{rendered.strip()}[/dim]")


@app.command(rich_help_panel=PANEL_LIBRARY)
def where(
    name: Annotated[str, typer.Argument(help="Part of an asset's name or path.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Results to show.")] = 20,
) -> None:
    """Show where an asset is stored.

    A direct answer to "I know I downloaded this — where did it go?".
    """
    with session_scope() as session:
        matches = InventoryEngine(session).locate(name, limit=limit)

        if not matches:
            console.print(f"[dim]Nothing matching {name!r} in the catalogue.[/dim]")
            return

        table = Table(box=None, pad_edge=False, header_style="bold")
        # Every column is bounded. An unbounded no-wrap column demands its full natural
        # width, and Rich makes room by dropping other columns — losing Name, which is
        # the one column that must always survive.
        table.add_column("Name", style="cyan", min_width=12, max_width=34,
                         overflow="ellipsis", no_wrap=True)
        table.add_column("Category", max_width=18, overflow="ellipsis", no_wrap=True)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Location", max_width=60, overflow="ellipsis", no_wrap=True)

        for item in matches:
            table.add_row(
                item.name,
                item.category_label,
                format_bytes(item.size_bytes),
                shorten_path(item.path, 60),
            )
        console.print(table)


@app.command(rich_help_panel=PANEL_LIBRARY)
def stats() -> None:
    """Summarise the catalogue."""
    with session_scope() as session:
        service = AssetService(session)
        counts = service.counts_by_kind()

        if not counts:
            console.print("[dim]Catalogue is empty. Run 'aam scan' first.[/dim]")
            return

        total = service.total_size()
        physical = service.total_size(physical=True)

        summary = Table(box=None, pad_edge=False, show_header=False)
        summary.add_column(style="bold")
        summary.add_column(justify="right")
        for kind, count in sorted(counts.items(), key=lambda item: -item[1]):
            summary.add_row(kind.title(), f"{count:,}")
        summary.add_row("", "")
        summary.add_row("Files", f"{service.file_count():,}")
        summary.add_row("Total size", format_bytes(total))
        if physical != total:
            summary.add_row("On disk", format_bytes(physical))
        console.print(Panel(summary, title="Catalogue", border_style="cyan", expand=False))

        for title, column in (("Storage by drive", "drive"), ("Storage by framework", "framework")):
            grouped = service.size_by(column)
            if not grouped:
                continue
            table = Table(box=None, pad_edge=False, title=title, title_justify="left")
            table.add_column(column.title())
            table.add_column("Size", justify="right")
            for key, size in list(grouped.items())[:12]:
                table.add_row(key, format_bytes(size))
            console.print(table)

        largest = service.largest(5)
        if largest:
            table = Table(box=None, pad_edge=False, title="Largest assets", title_justify="left")
            table.add_column("Name", style="cyan", max_width=46, overflow="ellipsis")
            table.add_column("Size", justify="right")
            for asset in largest:
                table.add_row(asset.display_name or asset.name, format_bytes(asset.size_bytes))
            console.print(table)


@app.command(rich_help_panel=PANEL_ABOUT)
def version(
    plugins: Annotated[
        bool, typer.Option("--plugins", help="Also list the loaded taxonomy plugins.")
    ] = False,
) -> None:
    """Show the version, where data is stored, and what the taxonomy loaded.

    The plugin count is here rather than buried in a debug flag because it is the one
    number that says whether this build can classify anything. A packaged binary that
    failed to bundle its plugins still runs, still prints tables, and quietly files every
    asset as "unclassified" - so it is worth being able to check in one command.
    """
    from ai_asset_manager.backend.taxonomy import default_registry

    settings = get_settings()
    registry = default_registry()

    console.print(f"[bold]AI Asset Manager[/bold] {__version__}")
    console.print(f"Database: {settings.db_path}")
    console.print(f"Data directory: {settings.data_dir}")
    console.print(
        f"Taxonomy: {len(registry.categories())} categories, "
        f"{len(registry.tasks())} tasks from {len(registry.plugins())} plugins"
    )

    if plugins:
        console.print()
        table = Table(box=None, pad_edge=False, header_style="bold")
        table.add_column("Plugin", style="cyan")
        table.add_column("Categories", justify="right")
        table.add_column("Provides", style="dim")

        for name in registry.plugins():
            owned = [
                category.id
                for category in registry.categories()
                if category.id.startswith(name) or name in category.id
            ]
            table.add_row(name, str(len(owned)), ", ".join(owned[:3]))
        console.print(table)


@app.command(rich_help_panel=PANEL_ABOUT)
def guide() -> None:
    """Show worked examples of the things people actually want to do."""
    recipes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
        ("Getting started", (
            ("aam scan --auto", "catalogue every cache found on this machine"),
            ("aam scan D:\\Models E:\\Datasets", "catalogue specific folders"),
            ("aam scan --add D:\\Models", "remember the folder for next time"),
            ("aam scan", "rescan everything remembered (skips unchanged)"),
        )),
        ("What do I have?", (
            ("aam inventory", "everything, by category"),
            ("aam inventory llm", "just the language models"),
            ("aam inventory ocr", "just the OCR models"),
            ("aam inventory datasets", "just the datasets"),
            ("aam inventory --tree", "the shape of the library"),
            ("aam inventory --details", "everything known about each asset"),
        )),
        ("Cutting it a different way", (
            ("aam inventory --group-by task", "grouped by what things are for"),
            ("aam inventory --group-by family", "grouped by model family"),
            ("aam inventory --domain medical", "one domain only"),
            ("aam inventory --drive F: --sort size", "biggest things on one drive"),
            ("aam inventory --storage", "where the space has gone"),
        )),
        ("Is any of it broken?", (
            ("aam inventory missing", "only what needs attention"),
            ("aam inventory health", "score and findings for everything"),
        )),
        ("Finding and exporting", (
            ("aam where qwen", "where did I put it?"),
            ("aam show 12", "one asset in full"),
            ("aam inventory --export csv -o library.csv", "take it to a spreadsheet"),
            ("aam inventory --export markdown", "paste it into notes"),
        )),
    )

    for heading, entries in recipes:
        console.print(f"\n[bold]{heading}[/bold]")
        table = Table(box=None, pad_edge=False, show_header=False, padding=(0, 2, 0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="dim")
        for command, description in entries:
            table.add_row(command, description)
        console.print(table)

    console.print(
        "\n[dim]Everything above is read-only. This tool never moves, renames or "
        "deletes a file.[/dim]"
    )


def main() -> None:
    """Entry point for the ``aam`` console script."""
    try:
        app()
    except KeyboardInterrupt:
        error_console.print("Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
