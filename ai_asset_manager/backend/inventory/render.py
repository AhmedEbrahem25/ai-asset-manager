"""Terminal rendering for inventory reports.

Kept out of the CLI module so the API and, later, the desktop UI can reuse the same
column choices and formatting decisions.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.panel import Panel
from rich.table import Table

from ai_asset_manager.backend.inventory.engine import (
    InventoryGroup,
    InventoryItem,
    InventoryReport,
)
from ai_asset_manager.backend.inventory.export import format_parameters
from ai_asset_manager.backend.utils.humanize import format_bytes, format_relative_time
from ai_asset_manager.backend.utils.paths import shorten_path

#: Width the Location column is shortened to. Paths are elided in the middle rather than
#: wrapped: a folded path spreads one row over fifteen lines and makes the table
#: unreadable, while the head and tail are what identify a location anyway.
LOCATION_WIDTH = 46

#: Colour per section, so models and datasets are distinguishable at a glance without
#: needing to read the category column.
SECTION_STYLES = {
    "models": "cyan",
    "datasets": "green",
    "papers": "magenta",
    "other": "white",
}


def render_table(items: Sequence[InventoryItem], *, show_details: bool = False) -> Table:
    """Render inventory items as a table.

    Args:
        items: The items to show.
        show_details: Include architecture, parameters and file counts.

    Returns:
        A Rich table.
    """
    # Every column truncates rather than wraps. A single over-long value would otherwise
    # fold its whole row over a dozen lines and destroy the alignment that makes a table
    # worth reading; an elided value still shows what it is.
    table = Table(box=None, pad_edge=False, header_style="bold", expand=False)
    table.add_column("Name", style="cyan", min_width=14, max_width=38,
                     overflow="ellipsis", no_wrap=True)
    table.add_column("Category", max_width=18, overflow="ellipsis", no_wrap=True)
    table.add_column("Framework", max_width=13, overflow="ellipsis", no_wrap=True)

    if show_details:
        table.add_column("Architecture", max_width=24, overflow="ellipsis", no_wrap=True)
        table.add_column("Params", justify="right", no_wrap=True)
        table.add_column("Quant", max_width=10, overflow="ellipsis", no_wrap=True)

    table.add_column("Format", max_width=12, overflow="ellipsis", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)

    if show_details:
        table.add_column("Files", justify="right", no_wrap=True)
        table.add_column("Modified", style="dim", max_width=14, overflow="ellipsis",
                         no_wrap=True)

    table.add_column(
        "Location", style="dim", max_width=LOCATION_WIDTH, no_wrap=True, overflow="ellipsis"
    )

    for item in items:
        row = [item.name, item.category_label, item.framework]

        if show_details:
            row.extend(
                [
                    item.architecture or "",
                    format_parameters(item),
                    item.quantization or "",
                ]
            )

        row.extend([item.format, format_bytes(item.size_bytes)])

        if show_details:
            row.extend(
                [
                    f"{item.file_count:,}",
                    format_relative_time(item.modified_at) if item.modified_at else "",
                ]
            )

        row.append(shorten_path(item.path, LOCATION_WIDTH))
        table.add_row(*row)

    return table


def render_dataset_table(items: Sequence[InventoryItem]) -> Table:
    """Render a dataset-specific table.

    Datasets are described by contents rather than by architecture, so they get their own
    columns; showing them a "Params" column would be all blanks.
    """
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("Name", style="green", min_width=14, max_width=34,
                     overflow="ellipsis", no_wrap=True)
    table.add_column("Type", max_width=20, overflow="ellipsis", no_wrap=True)
    table.add_column("Images", justify="right", no_wrap=True)
    table.add_column("Classes", justify="right", no_wrap=True)
    table.add_column("Splits", max_width=22, overflow="ellipsis", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column(
        "Location", style="dim", max_width=LOCATION_WIDTH, no_wrap=True, overflow="ellipsis"
    )

    for item in items:
        splits = ", ".join(f"{name}={count:,}" for name, count in item.splits.items())
        table.add_row(
            item.name,
            item.dataset_type or item.category_label,
            f"{item.num_images:,}" if item.num_images else "",
            str(item.num_classes) if item.num_classes else "",
            splits,
            format_bytes(item.size_bytes),
            shorten_path(item.path, LOCATION_WIDTH),
        )

    return table


def render_details(items: Sequence[InventoryItem]) -> Table:
    """Render items as per-asset detail blocks.

    The full detail set runs to fourteen fields. That is a record, not a table row: laid
    out horizontally the columns collapse to a few characters each and every value is
    elided. A key/value block per asset keeps all of it legible at any terminal width.
    """
    outer = Table(box=None, pad_edge=False, show_header=False, padding=(0, 0, 1, 0))
    outer.add_column()

    for item in items:
        block = Table(box=None, pad_edge=False, show_header=False)
        block.add_column(style="dim", width=14)
        block.add_column(overflow="fold")

        # The scanner's subkind usually restates the category in its own vocabulary
        # ("LLM · llm"), so it is only worth showing when it says something new.
        subcategory = (
            item.subcategory
            if item.subcategory and item.subcategory != item.category.value
            else None
        )

        rows: list[tuple[str, str]] = [
            ("Category", item.category_label + (f"  ·  {subcategory}" if subcategory else "")),
            ("Framework", item.framework),
            ("Format", item.format),
        ]

        if item.is_model:
            rows.extend(
                [
                    ("Architecture", item.architecture or "—"),
                    ("Parameters", format_parameters(item) or "—"),
                    ("Quantization", item.quantization or "—"),
                    ("Precision", item.precision or "—"),
                ]
            )
            if item.context_length:
                rows.append(("Context", f"{item.context_length:,}"))
            if item.repo_id:
                rows.append(("Repository", item.repo_id))
            if item.license:
                rows.append(("License", item.license))
        else:
            rows.append(("Dataset type", item.dataset_type or "—"))
            if item.num_images:
                rows.append(("Images", f"{item.num_images:,}"))
            if item.num_videos:
                rows.append(("Videos", f"{item.num_videos:,}"))
            if item.num_annotations:
                rows.append(("Annotations", f"{item.num_annotations:,}"))
            if item.num_classes:
                rows.append(("Classes", str(item.num_classes)))
            if item.splits:
                rows.append(
                    ("Splits", ", ".join(f"{k}={v:,}" for k, v in item.splits.items()))
                )

        rows.extend(
            [
                ("Size", format_bytes(item.size_bytes)),
                ("Files", f"{item.file_count:,}"),
                ("Drive", item.drive or "—"),
                (
                    "Modified",
                    format_relative_time(item.modified_at) if item.modified_at else "—",
                ),
                ("Root folder", item.root_folder),
                ("Path", item.path),
            ]
        )

        if item.tags:
            rows.append(("Tags", ", ".join(item.tags)))

        for label, value in rows:
            block.add_row(label, value)

        style = SECTION_STYLES.get(item.section.value, "white")
        outer.add_row(f"[bold {style}]{item.name}[/bold {style}]")
        outer.add_row(block)

    return outer


def render_summary(report: InventoryReport) -> Panel:
    """Render the headline summary panel."""
    summary = report.summary
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_column(justify="right", style="dim")

    for entry in summary.by_category:
        table.add_row(entry.label, f"{entry.count:,}", format_bytes(entry.total_bytes))

    if summary.by_category:
        table.add_row("", "", "")

    table.add_row("Total Assets", f"{summary.total_assets:,}", "")
    table.add_row("Total Storage", "", format_bytes(summary.total_bytes))
    if summary.physical_bytes and summary.physical_bytes != summary.total_bytes:
        table.add_row("On Disk", "", format_bytes(summary.physical_bytes))
    table.add_row("Files", f"{summary.total_files:,}", "")

    if summary.missing_assets:
        table.add_row("[yellow]Missing[/yellow]", f"{summary.missing_assets:,}", "")

    return Panel(table, title="AI Asset Inventory", border_style="cyan", expand=False)


def render_group_heading(group: InventoryGroup) -> str:
    """Return a formatted heading for a group."""
    return (
        f"\n[bold]{group.label}[/bold]  "
        f"[dim]{group.count} asset(s) · {format_bytes(group.total_bytes)}[/dim]"
    )


def render_storage_breakdown(report: InventoryReport, *, by: str = "drive") -> Table | None:
    """Render a storage breakdown by drive or framework."""
    data = report.summary.by_drive if by == "drive" else report.summary.by_framework
    if not data:
        return None

    table = Table(box=None, pad_edge=False, title=f"Storage by {by}", title_justify="left")
    table.add_column(by.title())
    table.add_column("Size", justify="right")
    for key, size in list(data.items())[:12]:
        table.add_row(key, format_bytes(size))
    return table
