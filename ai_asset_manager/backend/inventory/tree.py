"""Hierarchical view of an inventory.

A flat table answers "what do I have?"; a tree answers "what is my library *shaped* like?".
Section, then category, then family is the grouping that matches how people describe their
own storage — "under models I've got the OCR ones, and those are mostly PaddleOCR".

Nothing here re-queries anything. A tree is a regrouping of a report that has already been
built, which is why ``--tree`` costs nothing over the flat listing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ai_asset_manager.backend.inventory.categories import section_order
from ai_asset_manager.backend.inventory.engine import InventoryItem, InventoryReport

#: The default nesting. Family sits last because it is the level that most often collapses
#: — plenty of assets have no recognised family, and a level that is usually empty makes a
#: worse first split than a last one.
DEFAULT_LEVELS = ("section", "category", "family")

#: Levels a caller may nest by.
TREE_LEVELS = ("section", "category", "task", "domain", "family", "framework", "drive",
               "format")


@dataclass(slots=True)
class TreeNode:
    """One node of the inventory tree.

    A node is either a grouping (``children`` populated) or a leaf standing for a single
    asset (``item`` populated). Totals are held on every node so a collapsed branch can
    still report what it contains.
    """

    key: str
    label: str
    level: str = ""
    order: int = 0
    children: list[TreeNode] = field(default_factory=list)
    item: InventoryItem | None = None
    count: int = 0
    total_bytes: int = 0

    @property
    def is_leaf(self) -> bool:
        """Report whether this node stands for a single asset."""
        return self.item is not None


def build_tree(
    report: InventoryReport,
    *,
    levels: Sequence[str] = DEFAULT_LEVELS,
    label: str = "AI Library",
) -> TreeNode:
    """Group a report into a tree.

    Args:
        report: A report already built by the engine.
        levels: Nesting order, from :data:`TREE_LEVELS`.
        label: Text for the root node.

    Returns:
        The root :class:`TreeNode`.

    Levels that produce a single unnamed bucket are skipped rather than rendered as an
    empty rung - a tree of "Models / Unknown family / one asset" wastes two lines saying
    nothing.
    """
    wanted = [level for level in levels if level in TREE_LEVELS]
    root = TreeNode(key="root", label=label, level="root")
    for item in report.items:
        _insert(root, item, wanted)
    _tally(root)
    # The first two levels are the skeleton and are kept even when they hold one asset;
    # anything deeper that holds one asset is a rung that says what the asset's own line
    # already says.
    _collapse(root, protected=frozenset(wanted[:2]))
    _tally(root)
    _sort(root)
    return root


def _insert(node: TreeNode, item: InventoryItem, levels: Sequence[str]) -> None:
    """Place one item under a node, creating the intermediate levels it needs."""
    if not levels:
        node.children.append(
            TreeNode(
                key=str(item.asset_id),
                label=item.name,
                level="asset",
                item=item,
                count=1,
                total_bytes=item.size_bytes,
            )
        )
        return

    level, rest = levels[0], levels[1:]
    key, label, order = _bucket(item, level)

    if key is None:
        # This item has nothing to say at this level — an asset with no recognised family,
        # for instance. Skip the rung rather than inventing an "Unknown" one.
        _insert(node, item, rest)
        return

    child = next(
        (candidate for candidate in node.children
         if candidate.key == key and candidate.level == level),
        None,
    )
    if child is None:
        child = TreeNode(key=key, label=label, level=level, order=order)
        node.children.append(child)

    _insert(child, item, rest)


def _bucket(item: InventoryItem, level: str) -> tuple[str | None, str, int]:
    """Return ``(key, label, order)`` for an item at one nesting level."""
    resolvers: dict[str, Callable[[], tuple[str | None, str, int]]] = {
        "section": lambda: (item.section, item.section_label, section_order(item.section)),
        "category": lambda: (item.category, item.category_label, 0),
        "task": lambda: (item.task, item.task_label, 0),
        "domain": lambda: (item.domain, item.domain_label, 0),
        "family": lambda: (item.family, item.family or "", 0),
        "framework": lambda: (item.framework, item.framework.replace("_", " ").title(), 0),
        "drive": lambda: (item.drive, item.drive or "", 0),
        "format": lambda: (item.format, item.format.upper(), 0),
    }
    resolve = resolvers.get(level)
    if resolve is None:
        return None, "", 0
    key, label, order = resolve()
    return (key or None), (label or key or ""), order


def _tally(node: TreeNode) -> tuple[int, int]:
    """Fill in counts and sizes bottom-up."""
    if node.is_leaf:
        return node.count, node.total_bytes

    count = 0
    total = 0
    for child in node.children:
        child_count, child_bytes = _tally(child)
        count += child_count
        total += child_bytes

    node.count = count
    node.total_bytes = total
    return count, total


def _collapse(node: TreeNode, *, protected: frozenset[str]) -> None:
    """Hoist the children of single-asset grouping nodes at unprotected levels."""
    for child in node.children:
        _collapse(child, protected=protected)

    hoisted: list[TreeNode] = []
    for child in node.children:
        if (
            not child.is_leaf
            and child.level not in protected
            and child.level != "root"
            and child.count <= 1
        ):
            hoisted.extend(child.children)
        else:
            hoisted.append(child)
    node.children = hoisted


def _sort(node: TreeNode) -> None:
    """Order children by their declared position, then by size."""
    node.children.sort(key=lambda child: (child.order, -child.total_bytes, child.label.lower()))
    for child in node.children:
        _sort(child)


def flatten(node: TreeNode) -> list[InventoryItem]:
    """Return every asset beneath a node, depth first."""
    if node.is_leaf and node.item is not None:
        return [node.item]
    found: list[InventoryItem] = []
    for child in node.children:
        found.extend(flatten(child))
    return found
