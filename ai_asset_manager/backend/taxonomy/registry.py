"""The taxonomy registry and its plugin loader.

This module contains the *rules for holding* a taxonomy and none of the taxonomy itself.
Search it for the name of any model family, dataset layout or weight format and you will
find nothing: every category, task, domain, modality, classifier, health rule and statistic
arrives from a plugin under :mod:`ai_asset_manager.backend.taxonomy.plugins`. A test
asserts exactly that against this file's source, because it is the kind of property that
quietly stops holding the first time someone adds "just one" special case.

Adding an AI domain the project has never heard of — bioinformatics, seismology, whatever
comes next — means dropping one module into that package. It is picked up automatically,
and neither the scanner, the inventory engine, the database schema nor the CLI is touched.
Third-party distributions can do the same from outside the tree through the
``ai_asset_manager.taxonomy`` entry-point group.

Two decisions keep that promise honest:

*Forward references never fail.* A category may name a section that no plugin registered,
a task may name an unknown domain. The registry synthesises a descriptor rather than
raising, so plugin load order is irrelevant and a plugin can be installed without its
neighbours.

*Dispatch is by declared priority, not import order.* Classifiers are tried
highest-priority first and the first non-``None`` answer wins, mirroring the detector
registry the scanner already uses.
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ai_asset_manager.backend.taxonomy.types import (
    AssetProfile,
    Category,
    Classification,
    Classifier,
    ClassifierFunction,
    Domain,
    Finding,
    HealthReport,
    HealthRule,
    HealthRuleFunction,
    Modality,
    Section,
    StatisticFunction,
    StatisticProvider,
    Task,
)
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Entry-point group third-party packages publish plugins under. A distribution declaring
#: ``[project.entry-points."ai_asset_manager.taxonomy"] mine = "my_pkg.taxonomy"`` extends
#: the inventory without being part of this repository.
ENTRY_POINT_GROUP = "ai_asset_manager.taxonomy"

#: Category id used when nothing recognised an asset. Registered by the fallback plugin;
#: named here only so the registry has something to return when it must return something.
UNCLASSIFIED = "unclassified"


class TaxonomyRegistry:
    """Holds a taxonomy and dispatches classification, health and statistics over it."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._sections: dict[str, Section] = {}
        self._domains: dict[str, Domain] = {}
        self._tasks: dict[str, Task] = {}
        self._modalities: dict[str, Modality] = {}
        self._categories: dict[str, Category] = {}
        self._aliases: dict[str, tuple[str, ...]] = {}
        self._classifiers: list[Classifier] = []
        self._health_rules: list[HealthRule] = []
        self._statistics: list[StatisticProvider] = []
        self._loaded_plugins: list[str] = []
        self._sorted = True

    # -- registration -------------------------------------------------------

    def add_section(self, section: Section) -> Section:
        """Register a top-level section."""
        self._sections[section.id] = section
        return section

    def add_domain(self, domain: Domain) -> Domain:
        """Register a domain of AI work."""
        self._domains[domain.id] = domain
        return domain

    def add_task(self, task: Task) -> Task:
        """Register a task."""
        self._tasks[task.id] = task
        return task

    def add_modality(self, modality: Modality) -> Modality:
        """Register a data modality."""
        self._modalities[modality.id] = modality
        return modality

    def add_category(self, category: Category) -> Category:
        """Register a category and the aliases that select it.

        Re-registering an id replaces the previous definition, which is what lets a local
        plugin relabel or re-order a built-in category without forking it.
        """
        if category.id in self._categories:
            logger.debug("Category %r redefined", category.id)
        self._categories[category.id] = category
        for alias in (category.id, *category.aliases):
            self._aliases.setdefault(_normalise_alias(alias), (category.id,))
        return category

    def add_alias(self, name: str, categories: Sequence[str]) -> None:
        """Point one user-typed name at several categories.

        How ``vision`` comes to mean detection, segmentation, classification and tracking
        at once, without those categories having to know about each other.
        """
        self._aliases[_normalise_alias(name)] = tuple(categories)

    def add_classifier(
        self, function: ClassifierFunction, *, name: str, priority: int = 100
    ) -> Classifier:
        """Register a classification rule.

        Args:
            function: Takes a profile, returns a verdict or ``None`` to decline.
            name: Identifies the rule in logs and in ``--details`` output.
            priority: Higher runs first. See the plugin package docs for the bands.
        """
        classifier = Classifier(name=name, priority=priority, run=function)
        self._classifiers.append(classifier)
        self._sorted = False
        return classifier

    def add_health_rule(self, function: HealthRuleFunction, *, name: str) -> HealthRule:
        """Register a health rule."""
        rule = HealthRule(name=name, run=function)
        self._health_rules.append(rule)
        return rule

    def add_statistic(
        self, function: StatisticFunction, *, name: str
    ) -> StatisticProvider:
        """Register a statistics provider."""
        provider = StatisticProvider(name=name, run=function)
        self._statistics.append(provider)
        return provider

    # -- decorators ---------------------------------------------------------

    def classifier(
        self, *, priority: int = 100, name: str | None = None
    ) -> Callable[[ClassifierFunction], Classifier]:
        """Return a decorator registering a function as a classifier.

        The decorator form suits a plugin with one or two rules; a plugin with a dozen
        reads better registering them by name in ``register``, where the priorities line
        up in a single block.
        """

        def decorate(function: ClassifierFunction) -> Classifier:
            return self.add_classifier(
                function, name=name or function.__name__.lstrip("_"), priority=priority
            )

        return decorate

    def health_rule(
        self, *, name: str | None = None
    ) -> Callable[[HealthRuleFunction], HealthRule]:
        """Return a decorator registering a function as a health rule."""

        def decorate(function: HealthRuleFunction) -> HealthRule:
            return self.add_health_rule(
                function, name=name or function.__name__.lstrip("_")
            )

        return decorate

    def statistic(
        self, *, name: str | None = None
    ) -> Callable[[StatisticFunction], StatisticProvider]:
        """Return a decorator registering a function as a statistics provider."""

        def decorate(function: StatisticFunction) -> StatisticProvider:
            return self.add_statistic(
                function, name=name or function.__name__.lstrip("_")
            )

        return decorate

    # -- lookup -------------------------------------------------------------

    def section(self, section_id: str) -> Section:
        """Return a section, synthesising a placeholder for an unregistered id."""
        found = self._sections.get(section_id)
        if found is not None:
            return found
        return Section(id=section_id, label=_titleise(section_id), order=900)

    def domain(self, domain_id: str) -> Domain:
        """Return a domain, synthesising a placeholder for an unregistered id."""
        found = self._domains.get(domain_id)
        if found is not None:
            return found
        return Domain(id=domain_id, label=_titleise(domain_id), order=900)

    def task(self, task_id: str) -> Task:
        """Return a task, synthesising a placeholder for an unregistered id."""
        found = self._tasks.get(task_id)
        if found is not None:
            return found
        return Task(id=task_id, label=_titleise(task_id), order=900)

    def modality(self, modality_id: str) -> Modality:
        """Return a modality, synthesising a placeholder for an unregistered id."""
        found = self._modalities.get(modality_id)
        if found is not None:
            return found
        return Modality(id=modality_id, label=_titleise(modality_id), order=900)

    def category(self, category_id: str) -> Category:
        """Return a category, synthesising a placeholder for an unregistered id.

        A catalogue written by a newer version, or by a plugin since uninstalled, can hold
        category ids this registry has never seen. Reporting them under a readable label
        beats refusing to list the asset at all.
        """
        found = self._categories.get(category_id)
        if found is not None:
            return found
        return Category(
            id=category_id, label=_titleise(category_id), section="other", order=990
        )

    def categories(self, *, section: str | None = None) -> list[Category]:
        """Return every registered category in display order."""
        values = [
            category
            for category in self._categories.values()
            if section is None or category.section == section
        ]
        return sorted(values, key=lambda category: (category.order, category.label))

    def sections(self) -> list[Section]:
        """Return every registered section in display order."""
        return sorted(self._sections.values(), key=lambda section: section.order)

    def domains(self) -> list[Domain]:
        """Return every registered domain in display order."""
        return sorted(self._domains.values(), key=lambda domain: domain.order)

    def tasks(self, *, domain: str | None = None) -> list[Task]:
        """Return every registered task, optionally limited to one domain."""
        values = [
            task
            for task in self._tasks.values()
            if domain is None or task.domain == domain
        ]
        return sorted(values, key=lambda task: (task.order, task.label))

    def label_of(self, category_id: str) -> str:
        """Return a category's display label."""
        return self.category(category_id).label

    def section_of(self, category_id: str) -> str:
        """Return the section a category belongs to."""
        return self.category(category_id).section

    def order_of(self, category_id: str) -> int:
        """Return a category's display sort position."""
        return self.category(category_id).order

    def resolve_alias(self, name: str) -> tuple[str, ...] | None:
        """Resolve a user-typed selector to category ids.

        Explicit aliases win, then ``all``, then a section name, then a domain name. The
        last two are resolved live rather than stored, so a plugin that adds a category to
        the datasets section immediately widens ``aam inventory datasets`` without anyone
        having to remember to extend a list.

        Returns ``None`` for an unrecognised name so callers can list the valid options
        rather than silently returning an empty inventory.
        """
        key = _normalise_alias(name)

        explicit = self._aliases.get(key)
        if explicit is not None:
            return explicit

        if key == "all":
            return tuple(self._categories)

        in_section = tuple(category.id for category in self.categories(section=key))
        if in_section:
            return in_section

        in_domain = tuple(
            category.id
            for category in self.categories()
            if category.domain is not None and _normalise_alias(category.domain) == key
        )
        return in_domain or None

    def known_aliases(self) -> list[str]:
        """Return every accepted selector, for help text and error messages."""
        sections = {section.id for section in self._sections.values()}
        sections |= {category.section for category in self._categories.values()}
        domains = {
            category.domain
            for category in self._categories.values()
            if category.domain is not None
        }
        return sorted({"all", *self._aliases, *sections, *domains})

    def plugins(self) -> list[str]:
        """Return the names of the plugins that populated this registry."""
        return list(self._loaded_plugins)

    def classifiers(self) -> list[Classifier]:
        """Return the registered classifiers, highest priority first."""
        self._ensure_sorted()
        return list(self._classifiers)

    # -- dispatch -----------------------------------------------------------

    def classify(self, profile: AssetProfile) -> Classification:
        """Return the first classification a plugin offers for an asset.

        Falls back to :data:`UNCLASSIFIED` when nothing claims it, so callers always get a
        usable answer and an unrecognised asset still appears in the inventory.
        """
        self._ensure_sorted()
        for classifier in self._classifiers:
            try:
                result = classifier(profile)
            except Exception:
                # One misbehaving plugin must not blank the whole inventory. Log and let
                # the next classifier try.
                logger.exception(
                    "Classifier %r failed on %r", getattr(classifier, "name", "?"), profile.path
                )
                continue
            if result is not None:
                source = getattr(classifier, "name", "")
                return result if result.source else _with_source(result, source)
        return Classification(category=UNCLASSIFIED, evidence="nothing matched")

    def check_health(self, profile: AssetProfile) -> HealthReport:
        """Run every health rule against an asset and score the result.

        Reports ``evaluated=False`` when the file list was not loaded, rather than
        pronouncing a perfect score on evidence nobody fetched.
        """
        if not profile.files.loaded:
            return HealthReport(score=100, findings=(), evaluated=False)

        findings: list[Finding] = []
        for rule in self._health_rules:
            try:
                findings.extend(rule(profile))
            except Exception:
                logger.exception(
                    "Health rule %r failed on %r", getattr(rule, "name", "?"), profile.path
                )

        # Deduplicate by code: two plugins may reasonably both notice a missing licence,
        # and the asset should be penalised once.
        unique: dict[str, Finding] = {}
        for finding in findings:
            unique.setdefault(finding.code, finding)

        ordered = sorted(
            unique.values(), key=lambda finding: (-finding.severity.rank, finding.code)
        )
        score = max(0, 100 - sum(finding.penalty for finding in ordered))
        return HealthReport(score=score, findings=tuple(ordered), evaluated=True)

    def statistics(self, profile: AssetProfile) -> dict[str, Any]:
        """Merge every provider's statistics for an asset."""
        merged: dict[str, Any] = {}
        for provider in self._statistics:
            try:
                merged.update(provider(profile))
            except Exception:
                logger.exception(
                    "Statistic %r failed on %r", getattr(provider, "name", "?"), profile.path
                )
        return merged

    # -- loading ------------------------------------------------------------

    def load_plugins(self, *, extra: Iterable[str] = ()) -> None:
        """Import every built-in plugin, then any published by installed distributions."""
        from ai_asset_manager.backend.taxonomy import plugins as plugin_package

        modules = sorted(
            name
            for _, name, is_package in pkgutil.iter_modules(plugin_package.__path__)
            if not is_package and not name.startswith("_")
        )
        for name in modules:
            self._load_module(f"{plugin_package.__name__}.{name}", label=name)

        for name in extra:
            self._load_module(name, label=name)

        self._load_entry_points()
        self._ensure_sorted()
        logger.debug(
            "Taxonomy loaded: %d categor(y/ies) from %d plugin(s)",
            len(self._categories),
            len(self._loaded_plugins),
        )

    def _load_module(self, dotted: str, *, label: str) -> None:
        """Import one plugin module and let it register itself."""
        try:
            module = importlib.import_module(dotted)
        except Exception:
            # A broken third-party plugin should cost its own features and nothing else.
            logger.exception("Taxonomy plugin %r could not be imported", dotted)
            return

        register = getattr(module, "register", None)
        if callable(register):
            try:
                register(self)
            except Exception:
                logger.exception("Taxonomy plugin %r failed to register", dotted)
                return
        self._loaded_plugins.append(label)

    def _load_entry_points(self) -> None:
        """Load plugins published by other installed distributions."""
        from importlib.metadata import entry_points

        try:
            found = entry_points(group=ENTRY_POINT_GROUP)
        except Exception:  # pragma: no cover - depends on the installed metadata backend
            logger.debug("Entry-point discovery unavailable")
            return

        for entry in found:
            try:
                target = entry.load()
            except Exception:
                logger.exception("Taxonomy entry point %r could not be loaded", entry.name)
                continue
            register = getattr(target, "register", target)
            if callable(register):
                try:
                    register(self)
                except Exception:
                    logger.exception("Taxonomy entry point %r failed to register", entry.name)
                    continue
            self._loaded_plugins.append(entry.name)

    def _ensure_sorted(self) -> None:
        """Order classifiers by descending priority, stable within a band."""
        if not self._sorted:
            self._classifiers.sort(key=lambda item: -getattr(item, "priority", 0))
            self._sorted = True


_DEFAULT: TaxonomyRegistry | None = None
_LOCK = threading.Lock()


def default_registry() -> TaxonomyRegistry:
    """Return the process-wide registry, loading plugins on first use.

    Loading is deferred rather than done at import so that importing a model class does not
    drag in the whole taxonomy, and guarded by a lock because the watchdog indexer and an
    API request can reach it concurrently.
    """
    global _DEFAULT
    if _DEFAULT is None:
        with _LOCK:
            if _DEFAULT is None:
                registry = TaxonomyRegistry()
                registry.load_plugins()
                _DEFAULT = registry
    return _DEFAULT


def reset_default_registry() -> None:
    """Discard the cached registry so the next call reloads plugins.

    Exists for tests that install a plugin at runtime; production never calls it.
    """
    global _DEFAULT
    with _LOCK:
        _DEFAULT = None


def _with_source(classification: Classification, source: str) -> Classification:
    """Return a copy of a classification tagged with the classifier that produced it."""
    return Classification(
        category=classification.category,
        task=classification.task,
        domain=classification.domain,
        family=classification.family,
        modalities=classification.modalities,
        confidence=classification.confidence,
        evidence=classification.evidence,
        source=source,
    )


def _normalise_alias(name: str) -> str:
    """Normalise a user-typed selector so spelling variants all land on one key."""
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def _titleise(identifier: str) -> str:
    """Turn an unregistered id into a readable label."""
    return identifier.replace("_", " ").replace("-", " ").strip().title()
