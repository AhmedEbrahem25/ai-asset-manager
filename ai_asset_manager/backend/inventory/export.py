"""Inventory export.

Three formats behind one protocol. Adding PDF or Excel later means registering another
class here and nothing else — which is why the deferred formats from the original plan
cost nothing to leave out today.

Exports are pure functions of a report: they never touch the database or the filesystem
except to write the file the user asked for.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from ai_asset_manager.backend.inventory.engine import InventoryItem, InventoryReport
from ai_asset_manager.backend.utils.humanize import format_bytes, format_count

#: Columns written to CSV, in order. Sizes appear twice — raw bytes for spreadsheets to
#: sum and sort, and a human-readable form for reading — because a CSV that only has
#: "4.9 GiB" cannot be totalled and one that only has 5261334528 cannot be skimmed.
CSV_COLUMNS = (
    "name",
    "section",
    "category",
    "task",
    "domain",
    "family",
    "subcategory",
    "framework",
    "architecture",
    "format",
    "storage_format",
    "parameters",
    "parameters_exact",
    "quantization",
    "precision",
    "context_length",
    "dataset_type",
    "modalities",
    "size_bytes",
    "size_human",
    "file_count",
    "images",
    "videos",
    "annotations",
    "classes",
    "splits",
    "health_score",
    "health_status",
    "health_findings",
    "drive",
    "root_folder",
    "path",
    "last_modified",
    "repo_id",
    "license",
    "tags",
)


@runtime_checkable
class InventoryExporter(Protocol):
    """Renders an inventory report to text."""

    name: str
    extension: str

    def render(self, report: InventoryReport) -> str:
        """Return the rendered report."""
        ...


class CsvExporter:
    """Renders an inventory as CSV."""

    name = "csv"
    extension = "csv"

    def render(self, report: InventoryReport) -> str:
        """Return the report as CSV text."""
        buffer = io.StringIO()
        # QUOTE_MINIMAL with the default dialect: Windows paths contain backslashes,
        # which must not be treated as escapes, and names contain commas.
        writer = csv.DictWriter(
            buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for item in report.items:
            writer.writerow(_flat_row(item))
        return buffer.getvalue()


class JsonExporter:
    """Renders an inventory as JSON."""

    name = "json"
    extension = "json"

    def render(self, report: InventoryReport) -> str:
        """Return the report as indented JSON, summary included."""
        return json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str)


class MarkdownExporter:
    """Renders an inventory as Markdown."""

    name = "markdown"
    extension = "md"

    #: Table columns, chosen to stay readable at a typical document width.
    COLUMNS = ("Name", "Category", "Task", "Family", "Size", "Health", "Location")

    def render(self, report: InventoryReport) -> str:
        """Return the report as a Markdown document."""
        lines: list[str] = ["# AI Asset Inventory", ""]
        lines.append(f"_Generated {report.generated_at:%Y-%m-%d %H:%M UTC}_")
        lines.append("")

        lines.extend(self._summary_section(report))

        if report.groups:
            for group in report.groups:
                lines.append(f"## {group.label}")
                lines.append("")
                lines.append(
                    f"_{group.count} asset(s) · {format_bytes(group.total_bytes)}_"
                )
                lines.append("")
                lines.extend(self._table(group.items))
                lines.append("")
        else:
            lines.append("## Assets")
            lines.append("")
            lines.extend(self._table(report.items))
            lines.append("")

        return "\n".join(lines)

    def _summary_section(self, report: InventoryReport) -> list[str]:
        """Render the summary block."""
        summary = report.summary
        lines = ["## Summary", ""]
        for entry in summary.by_category:
            lines.append(
                f"- **{entry.label}**: {entry.count} "
                f"({format_bytes(entry.total_bytes)})"
            )
        lines.append("")
        lines.append(f"- **Total assets**: {summary.total_assets}")
        lines.append(f"- **Total storage**: {format_bytes(summary.total_bytes)}")
        if summary.physical_bytes and summary.physical_bytes != summary.total_bytes:
            lines.append(f"- **On disk**: {format_bytes(summary.physical_bytes)}")
        if summary.average_health is not None:
            lines.append(f"- **Average health**: {summary.average_health}/100")
        if summary.unhealthy_assets:
            lines.append(f"- **Need attention**: {summary.unhealthy_assets}")
        lines.append("")
        return lines

    def _table(self, items: Sequence[InventoryItem]) -> list[str]:
        """Render a Markdown table for a set of items."""
        if not items:
            return ["_No assets._"]

        lines = [
            "| " + " | ".join(self.COLUMNS) + " |",
            "|" + "|".join("---" for _ in self.COLUMNS) + "|",
        ]
        for item in items:
            score = item.health_score
            lines.append(
                "| "
                + " | ".join(
                    (
                        _escape_md(item.name),
                        item.category_label,
                        item.task_label or "—",
                        _escape_md(item.family or "—"),
                        format_bytes(item.size_bytes),
                        f"{score}/100" if score is not None else "—",
                        f"`{item.path}`",
                    )
                )
                + " |"
            )
        return lines


#: Every available exporter, keyed by the name a user types.
EXPORTERS: dict[str, InventoryExporter] = {
    "csv": CsvExporter(),
    "json": JsonExporter(),
    "md": MarkdownExporter(),
    "markdown": MarkdownExporter(),
}


def get_exporter(name: str) -> InventoryExporter | None:
    """Return an exporter by name, or ``None`` if the format is unknown."""
    return EXPORTERS.get(name.strip().lower())


def available_formats() -> list[str]:
    """Return the distinct export format names."""
    return ["csv", "json", "markdown"]


def export_report(report: InventoryReport, fmt: str, destination: Path | None = None) -> str:
    """Render a report and optionally write it to disk.

    Args:
        report: The inventory to render.
        fmt: ``csv``, ``json`` or ``markdown``.
        destination: File to write. When omitted, the rendered text is only returned.

    Returns:
        The rendered text.

    Raises:
        ValueError: If the format is not recognised.
    """
    exporter = get_exporter(fmt)
    if exporter is None:
        raise ValueError(
            f"Unknown export format {fmt!r}. Available: {', '.join(available_formats())}"
        )

    rendered = exporter.render(report)

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # newline="" keeps csv from doubling line endings on Windows; utf-8 keeps model
        # names in non-Latin scripts intact.
        destination.write_text(rendered, encoding="utf-8", newline="")

    return rendered


def suggest_filename(fmt: str, *, prefix: str = "inventory") -> str:
    """Return a sensible default filename for an export."""
    exporter = get_exporter(fmt)
    extension = exporter.extension if exporter else "txt"
    return f"{prefix}.{extension}"


def _flat_row(item: InventoryItem) -> dict[str, object]:
    """Flatten an item into CSV columns."""
    health = item.health
    findings = (
        "; ".join(finding.message for finding in health.findings)
        if health is not None and health.evaluated
        else ""
    )
    counts: dict[str, int] = item.stat("split_counts") or item.splits
    splits_text = (
        ";".join(f"{name}={count}" for name, count in counts.items())
        if counts
        else ";".join(item.stat("splits") or ())
    )

    return {
        "name": item.name,
        "section": item.section,
        "category": item.category_label,
        "task": item.task_label,
        "domain": item.domain_label,
        "family": item.family or "",
        "subcategory": item.subcategory or "",
        "framework": item.framework,
        "architecture": item.architecture or "",
        "format": item.format,
        "storage_format": item.stat("storage_format", "") or "",
        "parameters": item.param_count or "",
        "parameters_exact": "yes" if item.param_count_is_exact else "",
        "quantization": item.quantization or "",
        "precision": item.precision or "",
        "context_length": item.context_length or "",
        "dataset_type": item.dataset_type or "",
        "modalities": ";".join(item.modalities),
        "size_bytes": item.size_bytes,
        "size_human": format_bytes(item.size_bytes),
        "file_count": item.file_count,
        "images": item.num_images or "",
        "videos": item.num_videos or "",
        "annotations": item.num_annotations or int(item.stat("annotations", 0) or 0) or "",
        "classes": item.num_classes or "",
        "splits": splits_text,
        "health_score": item.health_score if item.health_score is not None else "",
        "health_status": item.health_status,
        "health_findings": findings,
        "drive": item.drive or "",
        "root_folder": item.root_folder,
        "path": item.path,
        "last_modified": item.modified_at.isoformat() if item.modified_at else "",
        "repo_id": item.repo_id or "",
        "license": item.license or "",
        "tags": ";".join(item.tags),
    }


def _escape_md(text: str) -> str:
    """Escape pipes so a name cannot break a Markdown table."""
    return text.replace("|", "\\|")


def format_parameters(item: InventoryItem) -> str:
    """Render a parameter count for display, marking estimates.

    Examples:
        A model whose count came from tensor shapes renders as ``8.2B``; one estimated
        from storage bytes renders as ``8.2B~`` so the two are never confused.
    """
    if not item.param_count:
        return ""
    suffix = "" if item.param_count_is_exact else "~"
    return f"{format_count(item.param_count)}{suffix}"
