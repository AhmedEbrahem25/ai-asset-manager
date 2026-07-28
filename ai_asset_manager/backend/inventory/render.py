"""Terminal rendering for inventory reports.

Kept out of the CLI module so the API and, later, the desktop UI can reuse the same column
choices and formatting decisions.

Two rules govern everything here. Columns truncate rather than wrap, because one over-long
value folding its row over a dozen lines destroys the alignment that makes a table worth
reading. And anything with more than about eight fields is rendered as a block of
key/value pairs rather than a row, because a fourteen-column table at eighty characters is
fourteen columns of ellipses.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from ai_asset_manager.backend.inventory.engine import (
    InventoryGroup,
    InventoryItem,
    InventoryReport,
)
from ai_asset_manager.backend.inventory.export import format_parameters
from ai_asset_manager.backend.inventory.tree import TreeNode
from ai_asset_manager.backend.models.enums import Severity
from ai_asset_manager.backend.utils.humanize import format_bytes, format_relative_time
from ai_asset_manager.backend.utils.paths import shorten_path

#: Width the Location column is shortened to. Paths are elided in the middle rather than
#: wrapped: a folded path spreads one row over fifteen lines and makes the table
#: unreadable, while the head and tail are what identify a location anyway.
LOCATION_WIDTH = 38

#: Minimum width of the label column in a breakdown table, wide enough that the longest
#: heading ("Storage by framework") fits on one line.
BREAKDOWN_LABEL_WIDTH = 16

#: Colour per section, so models and datasets are distinguishable at a glance without
#: needing to read the category column.
SECTION_STYLES = {
    "models": "cyan",
    "datasets": "green",
    "experiments": "yellow",
    "documents": "magenta",
    "other": "white",
}

#: Colour per health status.
HEALTH_STYLES = {
    "ok": "green",
    "warning": "yellow",
    "error": "red",
    "unknown": "dim",
}

#: Marker per finding severity, so a health listing reads without colour.
#: Deliberately ASCII: the Windows console defaults to cp1252, and a geometric-shape
#: character raises UnicodeEncodeError there rather than degrading to a box.
SEVERITY_MARKS = {
    Severity.ERROR: "[red]x[/red]",
    Severity.WARNING: "[yellow]![/yellow]",
    Severity.INFO: "[dim]-[/dim]",
}

#: Statistics worth putting in a detail block, in the order they read best. Anything a
#: plugin contributes that is not listed here still appears, after these — the list is a
#: preferred ordering, not a filter, so a new plugin's numbers show up without this module
#: being told about them.
STAT_ORDER = (
    "images", "videos", "audio_files", "annotations", "annotation_files", "classes",
    "splits", "split_counts", "storage_format", "version", "avg_image_bytes",
    "has_readme", "has_license", "parameters", "precision", "quantization",
    "context_length", "layers", "hidden_size", "vocab_size", "tensors", "shards",
    "tokenizer", "weight_formats", "base_model", "tracker", "checkpoints",
    "annotation_tool", "repo_id", "author", "license",
)

#: Statistics already shown elsewhere in a detail block, so showing them again is noise.
_STAT_SUPPRESSED = frozenset(
    {"files", "size_bytes", "on_disk_bytes", "parameters_exact", "class_names",
     "modalities", "top_extensions", "checkpoint_bytes", "revision", "pipeline_tag"}
)


def render_table(items: Sequence[InventoryItem], *, show_details: bool = False) -> Table:
    """Render inventory items as a table.

    Args:
        items: The items to show.
        show_details: Include architecture, parameters and file counts.

    Returns:
        A Rich table.
    """
    table = Table(box=None, pad_edge=False, header_style="bold", expand=False)
    table.add_column("Name", style="cyan", min_width=14, max_width=34,
                     overflow="ellipsis", no_wrap=True)
    table.add_column("Category", max_width=18, overflow="ellipsis", no_wrap=True)
    table.add_column("Task", max_width=20, overflow="ellipsis", no_wrap=True)

    if show_details:
        table.add_column("Family", max_width=14, overflow="ellipsis", no_wrap=True)
        table.add_column("Architecture", max_width=22, overflow="ellipsis", no_wrap=True)
        table.add_column("Params", justify="right", no_wrap=True)
        table.add_column("Quant", max_width=10, overflow="ellipsis", no_wrap=True)

    table.add_column("Framework", max_width=13, overflow="ellipsis", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)

    if show_details:
        table.add_column("Files", justify="right", no_wrap=True)
        table.add_column("Modified", style="dim", max_width=14, overflow="ellipsis",
                         no_wrap=True)

    table.add_column(
        "Location", style="dim", max_width=LOCATION_WIDTH, no_wrap=True, overflow="ellipsis"
    )

    for item in items:
        row = [item.name, item.category_label, item.task_label]

        if show_details:
            row.extend(
                [
                    item.family or "",
                    item.architecture or "",
                    format_parameters(item),
                    item.quantization or "",
                ]
            )

        row.extend([item.framework, format_bytes(item.size_bytes)])

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
    table.add_column("Name", style="green", min_width=14, max_width=30,
                     overflow="ellipsis", no_wrap=True)
    table.add_column("Task", max_width=20, overflow="ellipsis", no_wrap=True)
    table.add_column("Format", max_width=14, overflow="ellipsis", no_wrap=True)
    table.add_column("Samples", justify="right", no_wrap=True)
    table.add_column("Classes", justify="right", no_wrap=True)
    table.add_column("Splits", max_width=18, overflow="ellipsis", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Health", justify="right", no_wrap=True)
    table.add_column(
        "Location", style="dim", max_width=LOCATION_WIDTH, no_wrap=True, overflow="ellipsis"
    )

    for item in items:
        table.add_row(
            item.name,
            item.task_label or item.category_label,
            _dataset_format(item),
            _sample_count(item),
            str(item.num_classes) if item.num_classes else "",
            _splits_text(item),
            format_bytes(item.size_bytes),
            _health_cell(item),
            shorten_path(item.path, LOCATION_WIDTH),
        )

    return table


def render_details(items: Sequence[InventoryItem]) -> Table:
    """Render items as per-asset detail blocks.

    The full detail set runs past twenty fields once statistics and health are included.
    That is a record, not a table row: laid out horizontally the columns collapse to a few
    characters each and every value is elided. A key/value block per asset keeps all of it
    legible at any terminal width.
    """
    outer = Table(box=None, pad_edge=False, show_header=False, padding=(0, 0, 1, 0))
    outer.add_column()

    for item in items:
        block = Table(box=None, pad_edge=False, show_header=False)
        block.add_column(style="dim", width=14)
        block.add_column(overflow="fold")

        for label, value in _detail_rows(item):
            block.add_row(label, value)

        style = SECTION_STYLES.get(item.section, "white")
        heading = f"[bold {style}]{item.name}[/bold {style}]"
        if item.family:
            heading += f"  [dim]{item.family}[/dim]"
        outer.add_row(heading)
        outer.add_row(block)

    return outer


def _detail_rows(item: InventoryItem) -> list[tuple[str, str]]:
    """Return the key/value pairs describing one asset in full."""
    # The scanner's subkind usually restates the category in its own vocabulary
    # ("LLM · llm"), so it is only worth showing when it says something new.
    subcategory = (
        item.subcategory
        if item.subcategory and item.subcategory != item.category
        else None
    )

    rows: list[tuple[str, str]] = [
        ("Category", item.category_label + (f"  ·  {subcategory}" if subcategory else "")),
    ]

    if item.task_label:
        rows.append(("Task", item.task_label))
    if item.domain_label:
        rows.append(("Domain", item.domain_label))
    if item.family:
        rows.append(("Family", item.family))
    if item.modalities:
        rows.append(("Modalities", ", ".join(item.modalities)))

    rows.append(("Framework", item.framework))
    rows.append(("Format", item.format))

    if item.is_model:
        rows.append(("Architecture", item.architecture or "—"))

    for key in STAT_ORDER:
        if key in item.stats:
            rows.append((_stat_label(key), _stat_value(key, item.stats[key])))

    # Anything a plugin contributed that this module has never heard of still shows.
    for key in sorted(item.stats):
        if key not in STAT_ORDER and key not in _STAT_SUPPRESSED:
            rows.append((_stat_label(key), _stat_value(key, item.stats[key])))

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

    if item.health is not None and item.health.evaluated:
        rows.append(("Health", _health_cell(item)))
        for finding in item.health.findings:
            mark = SEVERITY_MARKS.get(finding.severity, "·")
            rows.append(("", f"{mark} {finding.message}"))

    if item.evidence:
        rows.append(("Identified by", f"[dim]{item.evidence}[/dim]"))

    return rows


def render_health(items: Sequence[InventoryItem]) -> Table:
    """Render a health listing: what is wrong, with what, and how to fix it."""
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("Name", style="cyan", min_width=14, max_width=30,
                     overflow="ellipsis", no_wrap=True)
    table.add_column("Category", max_width=18, overflow="ellipsis", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Findings", overflow="fold")

    for item in items:
        if item.health is None or not item.health.evaluated:
            continue

        lines = [
            f"{SEVERITY_MARKS.get(finding.severity, '·')} {finding.message}"
            for finding in item.health.findings
        ]
        table.add_row(
            item.name,
            item.category_label,
            _health_cell(item),
            "\n".join(lines) or "[green]nothing to report[/green]",
        )

    return table


def render_fixes(items: Sequence[InventoryItem], *, limit: int = 8) -> Table | None:
    """Render the distinct fix hints across a set of items, worst first.

    A health listing says what is wrong asset by asset; this says what to *do*, once per
    kind of problem, because twenty datasets missing a README need one instruction rather
    than twenty.
    """
    seen: dict[str, tuple[Severity, str, int]] = {}

    for item in items:
        if item.health is None:
            continue
        for finding in item.health.findings:
            if not finding.fix_hint:
                continue
            severity, hint, count = seen.get(
                finding.code, (finding.severity, finding.fix_hint, 0)
            )
            seen[finding.code] = (severity, hint, count + 1)

    if not seen:
        return None

    table = Table(box=None, pad_edge=False, title="What to do", title_justify="left",
                  header_style="bold")
    table.add_column("Affects", justify="right", no_wrap=True)
    table.add_column("Suggestion", overflow="fold")

    ranked = sorted(seen.items(), key=lambda pair: (-pair[1][0].rank, -pair[1][2]))
    for _, (severity, hint, count) in ranked[:limit]:
        mark = SEVERITY_MARKS.get(severity, "·")
        table.add_row(f"{count}x", f"{mark} {hint}")

    return table


def render_tree(root: TreeNode) -> Tree:
    """Render an inventory tree.

    Grouping nodes carry their subtotal so a collapsed branch still says how much is under
    it; leaves carry their own size and location.
    """
    tree = Tree(
        f"[bold]{root.label}[/bold]  "
        f"[dim]{root.count} asset(s) · {format_bytes(root.total_bytes)}[/dim]"
    )
    for child in root.children:
        _add_branch(tree, child)
    return tree


def _add_branch(parent: Tree, node: TreeNode) -> None:
    """Attach one node and its descendants to a Rich tree."""
    if node.is_leaf and node.item is not None:
        item = node.item
        detail = format_bytes(item.size_bytes)
        if item.health_status in ("warning", "error"):
            style = HEALTH_STYLES[item.health_status]
            detail += f" · [{style}]{item.health_score}/100[/{style}]"
        parent.add(f"{item.name}  [dim]{detail}[/dim]")
        return

    style = SECTION_STYLES.get(node.key, "bold") if node.level == "section" else "bold"
    branch = parent.add(
        f"[{style}]{node.label}[/{style}]  "
        f"[dim]{node.count} · {format_bytes(node.total_bytes)}[/dim]"
    )
    for child in node.children:
        _add_branch(branch, child)


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

    if summary.average_health is not None:
        style = _score_style(summary.average_health)
        table.add_row("Health", f"[{style}]{summary.average_health}/100[/{style}]", "")
    if summary.unhealthy_assets:
        table.add_row(
            "[yellow]Need attention[/yellow]", f"{summary.unhealthy_assets:,}", ""
        )
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

    # The label column is given a floor because Rich wraps a table's title to the table's
    # width: a two-column table of short drive letters is narrower than "Storage by
    # framework", and the heading would fold in half.
    table = Table(box=None, pad_edge=False, title=f"Storage by {by}", title_justify="left")
    table.add_column(by.title(), min_width=BREAKDOWN_LABEL_WIDTH)
    table.add_column("Size", justify="right")
    for key, size in list(data.items())[:12]:
        table.add_row(key, format_bytes(size))
    return table


def render_distribution(report: InventoryReport, *, by: str = "task") -> Table | None:
    """Render how many assets fall under each task, domain or family."""
    source = {
        "task": report.summary.by_task,
        "domain": report.summary.by_domain,
        "family": report.summary.by_family,
    }.get(by)
    if not source:
        return None

    from ai_asset_manager.backend.inventory.categories import domain_label, task_label

    table = Table(box=None, pad_edge=False, title=f"Assets by {by}", title_justify="left")
    table.add_column(by.title(), min_width=BREAKDOWN_LABEL_WIDTH)
    table.add_column("Count", justify="right")

    for key, count in list(source.items())[:15]:
        if by == "task":
            label = task_label(key)
        elif by == "domain":
            label = domain_label(key)
        else:
            label = key
        table.add_row(label or key, str(count))

    return table


# -- cell formatting --------------------------------------------------------


def _dataset_format(item: InventoryItem) -> str:
    """Return the dataset's layout and, when it differs, how it is stored."""
    layout = (item.dataset_type or "").replace("_", " ")
    storage = item.stat("storage_format")
    if storage and storage != item.dataset_type:
        return f"{layout or storage}" if not layout else f"{layout} / {storage}"
    return layout or (storage or "")


def _sample_count(item: InventoryItem) -> str:
    """Return the most meaningful sample count for a dataset."""
    for value in (item.num_images, item.num_videos, item.stat("audio_files", 0)):
        if value:
            return f"{int(value):,}"
    annotations = item.stat("annotations", 0)
    return f"{int(annotations):,}" if annotations else ""


def _splits_text(item: InventoryItem) -> str:
    """Return a compact rendering of a dataset's splits."""
    counts = item.stat("split_counts") or item.splits
    if counts:
        return ", ".join(f"{name}={count:,}" for name, count in counts.items())
    names = item.stat("splits")
    return ", ".join(names) if names else ""


def _health_cell(item: InventoryItem) -> str:
    """Return a coloured health score, or a dash when health was not evaluated."""
    score = item.health_score
    if score is None:
        return "[dim]—[/dim]"
    style = _score_style(score)
    return f"[{style}]{score}/100[/{style}]"


def _score_style(score: int) -> str:
    """Return the colour a health score should be shown in."""
    if score >= 90:
        return "green"
    if score >= 70:
        return "yellow"
    return "red"


def _stat_label(key: str) -> str:
    """Turn a statistic key into a display label."""
    return key.replace("_", " ").capitalize()


def _stat_value(key: str, value: object) -> str:
    """Format one statistic for display."""
    if isinstance(value, bool):
        return "yes" if value else "[yellow]no[/yellow]"
    if key.endswith("_bytes"):
        return format_bytes(int(value)) if isinstance(value, int) else str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, dict):
        return ", ".join(f"{name}={count}" for name, count in value.items())
    if isinstance(value, list | tuple):
        return ", ".join(str(entry) for entry in value)
    return str(value)
