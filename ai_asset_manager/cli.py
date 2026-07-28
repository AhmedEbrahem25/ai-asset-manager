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
from typing import Annotated

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

app = typer.Typer(
    name="aam",
    help="AI Asset Manager — catalogue, search and clean up local AI models and datasets.",
    no_args_is_help=True,
    add_completion=False,
)
roots_app = typer.Typer(help="Manage the folders that get scanned.", no_args_is_help=True)
app.add_typer(roots_app, name="roots")

# Windows consoles default to a legacy code page, and model names routinely contain
# characters it cannot encode. Rich handles the encoding; forcing it here means a name
# with CJK or emoji in it prints as replacement characters instead of aborting the
# command with a UnicodeEncodeError.
console = Console(soft_wrap=False)
error_console = Console(stderr=True, style="bold red")


def _bootstrap(*, verbose: bool = False, database: Path | None = None) -> None:
    """Configure logging and ensure the database exists."""
    settings = get_settings()
    configure_logging(
        "DEBUG" if verbose else settings.log_level,
        log_file=settings.log_file,
    )
    if database is not None:
        engine = configure_engine(f"sqlite+pysqlite:///{database.as_posix()}")
    else:
        settings.ensure_data_dir()
        engine = get_engine()
    init_database(engine)


@app.callback()
def main_callback(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
    database: Annotated[
        Path | None, typer.Option("--database", help="Use a specific SQLite file.")
    ] = None,
) -> None:
    """Set up logging and the database before any command runs."""
    _bootstrap(verbose=verbose, database=database)


@app.command()
def version() -> None:
    """Show the version and where data is stored."""
    settings = get_settings()
    console.print(f"[bold]AI Asset Manager[/bold] {__version__}")
    console.print(f"Database: {settings.db_path}")
    console.print(f"Data directory: {settings.data_dir}")


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


@app.command()
def scan(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Folders to scan. Defaults to the registered roots."),
    ] = None,
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
    """Scan folders and update the catalogue.

    Rescans are incremental: assets whose files are unchanged are skipped entirely.
    """
    targets = [str(path) for path in paths] if paths else None

    with session_scope() as session:
        service = ScanService(session)

        if add and targets:
            for target in targets:
                service.add_root(target)
            session.commit()

        if not targets and not service.list_roots(enabled_only=True):
            error_console.print(
                "No scan roots configured. Pass folders directly, or add them with "
                "'aam roots add <path>'."
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


@app.command("list")
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


@app.command()
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


@app.command()
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


def main() -> None:
    """Entry point for the ``aam`` console script."""
    try:
        app()
    except KeyboardInterrupt:
        error_console.print("Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
