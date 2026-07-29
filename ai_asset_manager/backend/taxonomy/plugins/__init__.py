"""Taxonomy plugins.

Every category, task, domain, modality, classifier, health rule and statistic the
inventory knows lives in one of these modules. Nothing here is referenced by name from the
core: the registry imports whatever it finds, so adding a module adds its knowledge.

Writing one
-----------

A plugin is a module exposing ``register(registry)``::

    from ai_asset_manager.backend.taxonomy import AssetProfile, Classification
    from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry

    def register(registry: TaxonomyRegistry) -> None:
        registry.add_domain(Domain(id="bioinformatics", label="Bioinformatics"))
        registry.add_category(Category(
            id="genomics_dataset", label="Genomics Dataset",
            section="datasets", order=290, domain="bioinformatics",
            aliases=("genomics", "genomics-datasets"),
        ))

        @registry.classifier(priority=600, name="genomics-dataset")
        def _genomics(profile: AssetProfile) -> Classification | None:
            if profile.files.count(".fastq", ".bam", ".vcf") > 0:
                return Classification(
                    category="genomics_dataset", task="variant_calling",
                    domain="bioinformatics", evidence="sequencing files present",
                )
            return None

Drop that file in this package and ``aam inventory genomics`` works. The scanner, the
inventory engine, the database schema and the CLI are untouched. Distributions outside
this repository do the same through the ``ai_asset_manager.taxonomy`` entry-point group.

Conventions
-----------

*Priority bands.* Classifiers are tried in descending priority. Leave gaps so a rule can
be inserted without renumbering its neighbours.

===========  ==================================================================
900 - 999    Structural certainties — the catalogue already said what this is.
700 - 899    Task-specific models whose evidence would otherwise be misread by a
             broader rule. OCR sits here because an OCR model's architecture ends
             in ``ForConditionalGeneration`` and a language rule would claim it.
400 - 699    Ordinary domain rules.
100 - 399    Name-only guesses, for bare weight files with no configuration.
below 0      The fallback. Only ``core`` should register here.
===========  ==================================================================

*Return ``None`` freely.* A classifier that does not recognise an asset must decline so
the next one gets a turn. Returning a low-confidence guess starves better-informed rules.

*Read only what you were given.* An :class:`~ai_asset_manager.backend.taxonomy.AssetProfile`
holds everything the scanner recorded, including the asset's file list. There is no
session and no filesystem access, so a plugin cannot make the inventory slow or unsafe.
"""

from __future__ import annotations
