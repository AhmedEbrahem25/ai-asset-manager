r"""Deriving the relationship graph from what the scan already found.

Every rule here works from the catalogue alone — paths, kinds and the metadata the parsers
already extracted. Nothing re-reads the disk, which is what lets the graph be rebuilt after
any scan for the cost of a couple of queries.

The rules are ordered by how certain they are, and it is worth being explicit about that
because the graph is only useful if a strong edge and a guess look different:

*Containment is a fact.* A checkpoint whose path is inside a project's directory belongs to
that project. There is nothing to infer.

*Production is nearly a fact.* Ultralytics writes ``weights/best.pt`` inside the run
directory; a weight file inside a run was produced by it.

*Ancestry is a guess.* An adapter naming ``Qwen/Qwen2.5-7B`` as its base model is telling
the truth about what it patches, but whether the copy on this disk is that model is an
inference from a name, and it is recorded as one.
"""

from __future__ import annotations

import os
from collections import defaultdict

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ai_asset_manager.backend.models import Asset, AssetLink, LinkRelation, ModelDetails
from ai_asset_manager.backend.models.enums import AssetKind
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Kinds that contain other assets rather than being contained by them.
CONTAINER_KINDS = (AssetKind.PROJECT, AssetKind.EXPERIMENT, AssetKind.ANNOTATION_PROJECT)

#: Kinds that a training run can be said to have produced.
PRODUCED_KINDS = (AssetKind.CHECKPOINT, AssetKind.MODEL, AssetKind.ADAPTER)


class LinkingService:
    """Rebuilds the asset relationship graph."""

    def __init__(self, session: Session) -> None:
        """Bind the service to a session."""
        self.session = session

    def rebuild(self) -> int:
        """Recompute every derived edge, replacing what was there.

        Returns:
            The number of edges in the rebuilt graph.
        """
        assets = list(self.session.scalars(select(Asset).where(~Asset.is_missing)))
        edges = self._derive(assets)

        self.session.execute(delete(AssetLink))
        self.session.add_all(edges)
        self.session.flush()

        logger.info("Rebuilt relationship graph: %d edge(s)", len(edges))
        return len(edges)

    def _derive(self, assets: list[Asset]) -> list[AssetLink]:
        """Return every edge implied by this set of assets."""
        seen: set[tuple[int, int, str]] = set()
        edges: list[AssetLink] = []

        def add(source: Asset, target: Asset, relation: LinkRelation,
                confidence: float, rule: str) -> None:
            if source.id == target.id:
                return
            key = (source.id, target.id, relation.value)
            if key in seen:
                return
            seen.add(key)
            edges.append(
                AssetLink(
                    source_id=source.id, target_id=target.id, relation=relation.value,
                    confidence=confidence, derived_by=rule,
                )
            )

        containers = {
            kind: [a for a in assets if a.kind == kind] for kind in CONTAINER_KINDS
        }

        for asset in assets:
            run = _innermost(asset, containers[AssetKind.EXPERIMENT])
            if run is not None and asset.kind in PRODUCED_KINDS:
                add(asset, run, LinkRelation.PRODUCED_BY, 1.0, "containment.run")

            project = _innermost(asset, containers[AssetKind.PROJECT])
            if project is not None:
                add(asset, project, LinkRelation.BELONGS_TO, 1.0, "containment.project")

            labelling = _innermost(asset, containers[AssetKind.ANNOTATION_PROJECT])
            if labelling is not None and asset.kind is AssetKind.DATASET:
                add(asset, labelling, LinkRelation.BELONGS_TO, 1.0, "containment.labelling")

        edges.extend(self._base_model_edges(assets, seen))
        return edges

    def _base_model_edges(
        self, assets: list[Asset], seen: set[tuple[int, int, str]]
    ) -> list[AssetLink]:
        """Link adapters and derivatives to the base model they name.

        A ``base_model`` field says ``Qwen/Qwen2.5-0.5B-Instruct``. Whether the copy on
        this disk is that model can only be judged by name, so matching is on the repo id
        and, failing that, on the trailing path segment — and the resulting edge says it is
        a guess.
        """
        by_repo: dict[str, Asset] = {}
        by_name: dict[str, list[Asset]] = defaultdict(list)
        for asset in assets:
            known: ModelDetails | None = asset.model_details
            if known is not None and known.repo_id:
                by_repo.setdefault(known.repo_id.lower(), asset)
            by_name[asset.name.lower()].append(asset)

        edges: list[AssetLink] = []
        for asset in assets:
            details = asset.model_details
            if details is None or not details.base_model:
                continue

            declared = details.base_model.strip().lower()
            target = by_repo.get(declared)
            confidence = 0.9
            if target is None:
                tail = declared.rsplit("/", 1)[-1]
                matches = by_name.get(tail, [])
                # One match is an answer; several is an ambiguity, and picking arbitrarily
                # would put a confident-looking wrong edge in the graph.
                if len(matches) != 1:
                    continue
                target = matches[0]
                confidence = 0.6

            relation = (
                LinkRelation.ADAPTS
                if asset.kind is AssetKind.ADAPTER
                else LinkRelation.DERIVED_FROM
            )
            key = (asset.id, target.id, relation.value)
            if asset.id == target.id or key in seen:
                continue
            seen.add(key)
            edges.append(
                AssetLink(
                    source_id=asset.id, target_id=target.id, relation=relation.value,
                    confidence=confidence, derived_by="metadata.base_model",
                )
            )
        return edges


def _innermost(asset: Asset, containers: list[Asset]) -> Asset | None:
    """Return the deepest container holding this asset, if any.

    Deepest rather than first: a repository may hold a project inside a project — on the
    development machine ``thorn-nlp`` contains ``services/clause_detector`` — and the
    nearer one is the meaningful owner of anything between them.
    """
    best: Asset | None = None
    path = _key(asset.root_path)
    for container in containers:
        root = _key(container.root_path)
        if path == root:
            continue
        if path.startswith(root.rstrip("/") + "/") and (
            best is None or len(root) > len(_key(best.root_path))
        ):
            best = container
    return best


def _key(path: str) -> str:
    """Return a path in a form safe to compare with ``startswith``."""
    normalised = path.replace("\\", "/")
    return normalised.lower() if os.name == "nt" else normalised
