#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8.1",
#   "huggingface_hub>=0.24",
#   "datasets>=2.14",
# ]
# ///
"""Reference PyAgent external tool: search Hugging Face Hub datasets.

Drop this file into ``~/.pyagent/tools/`` and run ``/tools reload`` in
PyAgent to register a ``search_hf_datasets`` tool with the agent. The
``# /// script`` block above lets ``uv`` install the heavy
``huggingface_hub`` and ``datasets`` deps in an isolated venv on first
invocation, so the core PyAgent install stays lean.

Tool contract:
- ``uv run search_hf_datasets.py describe``
    Prints a JSON manifest (name, description, JSON-Schema parameters).
- ``uv run search_hf_datasets.py invoke --args-file <path>``
    Reads a JSON object from ``<path>`` and prints the markdown report
    to stdout. Non-zero exit + stderr signals an error to PyAgent.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import click


MAX_RESULTS = 50
MAX_SAMPLE_CHARS = 500
DEFAULT_SAMPLE_ROWS = 3
DEFAULT_MAX_ENRICHED = 3
DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0
TOOL_TIMEOUT_ENV_VAR = "PYAGENT_HF_TOOL_TIMEOUT"
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

_HF_HTTP_TIMEOUT_DEFAULTS: dict[str, str] = {
    "HF_HUB_DOWNLOAD_TIMEOUT": "20",
    "HF_HUB_ETAG_TIMEOUT": "10",
}

VALID_SORT_FIELDS: tuple[str, ...] = (
    "downloads",
    "likes",
    "lastModified",
    "createdAt",
    "trendingScore",
)
SORT_ALIASES: dict[str, str] = {
    "trending": "trendingScore",
    "last_modified": "lastModified",
    "created_at": "createdAt",
}


def _resolve_hf_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for name in HF_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_sort(sort_by: str) -> str:
    candidate = SORT_ALIASES.get(sort_by, sort_by)
    if candidate not in VALID_SORT_FIELDS:
        return "downloads"
    return candidate


def _build_filter_tags(
    license_: list[str] | None,
    modality: list[str] | None,
    extra_tags: list[str] | None,
) -> list[str]:
    tags: list[str] = []
    for value in license_ or []:
        tags.append(f"license:{value}")
    for value in modality or []:
        tags.append(f"modality:{value}")
    if extra_tags:
        tags.extend(extra_tags)
    return tags


def _extract_metadata_from_tags(tags: list[str]) -> dict[str, list[str]]:
    metadata: dict[str, list[str]] = {
        "languages": [],
        "task_categories": [],
        "modalities": [],
        "licenses": [],
    }
    for tag in tags:
        if not isinstance(tag, str) or ":" not in tag:
            continue
        prefix, _, value = tag.partition(":")
        if prefix == "language":
            metadata["languages"].append(value)
        elif prefix == "task_categories":
            metadata["task_categories"].append(value)
        elif prefix == "modality":
            metadata["modalities"].append(value)
        elif prefix == "license":
            metadata["licenses"].append(value)
    return metadata


def _clean_description(description: str | None) -> str:
    if not description:
        return ""
    cleaned = description.replace("\t", " ")
    return " ".join(cleaned.split())


@dataclass
class DatasetReport:
    id: str
    author: str = "unknown"
    description: str = ""
    license: str = "unknown"
    tags: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    task_categories: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    last_modified: str | None = None
    configs: list[str] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=dict)
    sample_schema: dict[str, str] = field(default_factory=dict)
    sample_rows: list[dict] = field(default_factory=list)
    sample_error: str | None = None

    def to_markdown(self, include_samples: bool = True) -> str:
        lines = [f"## `{self.id}`"]

        description = _clean_description(self.description)
        if description:
            truncated = description[:240]
            if len(description) > 240:
                truncated = f"{truncated}..."
            lines.append("")
            lines.append(f"**Description**: {truncated}")

        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| **Author** | {self.author} |")
        lines.append(f"| **License** | `{self.license}` |")
        lines.append(f"| **Downloads** | {self.downloads:,} |")
        if self.likes:
            lines.append(f"| **Likes** | {self.likes:,} |")
        if self.last_modified:
            lines.append(f"| **Last Modified** | {self.last_modified} |")

        if self.languages:
            lines.append(
                f"| **Languages** | {', '.join(self.languages[:8])}{' ...' if len(self.languages) > 8 else ''} |"
            )
        if self.task_categories:
            lines.append(
                f"| **Tasks** | {', '.join(self.task_categories[:8])}{' ...' if len(self.task_categories) > 8 else ''} |"
            )
        if self.modalities:
            lines.append(f"| **Modalities** | {', '.join(self.modalities)} |")
        if self.tags:
            tag_preview = ", ".join(self.tags[:10])
            suffix = f" ... (+{len(self.tags) - 10} more)" if len(self.tags) > 10 else ""
            lines.append(f"| **Tags (preview)** | {tag_preview}{suffix} |")

        if self.configs:
            lines.append("")
            lines.append("### Configurations")
            for config in self.configs[:5]:
                splits = self.splits.get(config, [])
                splits_text = ", ".join(splits) if splits else "_unknown_"
                lines.append(f"- `{config}`: splits = {splits_text}")
            if len(self.configs) > 5:
                lines.append(
                    f"- _... and {len(self.configs) - 5} more configs_")

        if include_samples and self.sample_schema:
            lines.append("")
            lines.append("### Schema Preview")
            lines.append("| Column | Type |")
            lines.append("|--------|------|")
            for column, dtype in list(self.sample_schema.items())[:12]:
                lines.append(f"| `{column}` | `{dtype}` |")
            if len(self.sample_schema) > 12:
                lines.append(
                    f"| _..._ | _{len(self.sample_schema) - 12} more columns_ |")

        if include_samples and self.sample_rows:
            lines.append("")
            lines.append(
                f"### Sample Data (first {len(self.sample_rows)} row{'s' if len(self.sample_rows) != 1 else ''})"
            )
            lines.append("```json")
            for index, row in enumerate(self.sample_rows, start=1):
                truncated_row = {
                    key: (
                        f"{value[:MAX_SAMPLE_CHARS]}..."
                        if isinstance(value, str) and len(value) > MAX_SAMPLE_CHARS
                        else value
                    )
                    for key, value in row.items()
                }
                lines.append(
                    f"// Row {index}: {json.dumps(truncated_row, ensure_ascii=False, default=str)}"
                )
            lines.append("```")

        if include_samples and self.sample_error:
            lines.append("")
            lines.append(
                f"_Schema/sample fetch unavailable: {self.sample_error}_")

        return "\n".join(lines)


def _format_last_modified(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _populate_report_from_dataset_info(ds: Any) -> DatasetReport:
    tags = list(ds.tags or [])
    report = DatasetReport(
        id=getattr(ds, "id", "<unknown>"),
        author=getattr(ds, "author", None) or "unknown",
        description=getattr(ds, "description", "") or "",
        tags=tags,
        downloads=getattr(ds, "downloads", 0) or 0,
        likes=getattr(ds, "likes", 0) or 0,
        last_modified=_format_last_modified(
            getattr(ds, "last_modified", None)),
    )
    metadata = _extract_metadata_from_tags(tags)
    report.languages = metadata["languages"]
    report.task_categories = metadata["task_categories"]
    report.modalities = metadata["modalities"]
    if metadata["licenses"]:
        report.license = metadata["licenses"][0]
    return report


def _enrich_report_with_samples(
    report: DatasetReport,
    *,
    hf_token: str | None,
    sample_rows: int,
) -> None:
    try:
        from datasets import (  # type: ignore[import-not-found]
            get_dataset_config_names,
            get_dataset_split_names,
            load_dataset,
        )
    except Exception as exc:
        report.sample_error = f"`datasets` package unavailable: {exc}"
        return

    try:
        configs = get_dataset_config_names(report.id, token=hf_token)
    except Exception as exc:
        report.sample_error = f"could not list configs ({type(exc).__name__}: {exc})"
        return

    report.configs = list(configs)

    for config in report.configs[:3]:
        try:
            report.splits[config] = list(
                get_dataset_split_names(
                    report.id, config_name=config, token=hf_token)
            )
        except Exception:
            continue

    if not report.configs:
        return

    primary_config = report.configs[0]
    primary_split: str | None = None
    if report.splits.get(primary_config):
        primary_split = report.splits[primary_config][0]

    try:
        ds_obj = load_dataset(
            report.id,
            name=primary_config,
            split=primary_split,
            token=hf_token,
            streaming=True,
        )
    except Exception as exc:
        report.sample_error = (
            f"streaming load failed for `{primary_config}` ({type(exc).__name__}: {exc})"
        )
        return

    features = getattr(ds_obj, "features", None)
    if features:
        report.sample_schema = {
            column: (str(getattr(value, "dtype", value)) or str(value))
            for column, value in features.items()
        }

    try:
        iterator = iter(ds_obj)
        for _ in range(sample_rows):
            try:
                report.sample_rows.append(next(iterator))
            except StopIteration:
                break
    except Exception as exc:
        report.sample_error = f"sample iteration failed ({type(exc).__name__}: {exc})"


def _set_default_hf_http_timeouts() -> None:
    for name, default in _HF_HTTP_TIMEOUT_DEFAULTS.items():
        os.environ.setdefault(name, default)


def _resolve_tool_timeout(explicit: float | int | None) -> float:
    if explicit is not None:
        try:
            value = float(explicit)
        except (TypeError, ValueError):
            return DEFAULT_TOOL_TIMEOUT_SECONDS
        return max(1.0, value)

    raw = os.environ.get(TOOL_TIMEOUT_ENV_VAR)
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_TOOL_TIMEOUT_SECONDS


def _run_with_timeout(
    func: Callable[[], str],
    timeout: float,
) -> tuple[str | None, bool, BaseException | None]:
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["result"] = func()
        except BaseException as exc:  # noqa: BLE001 - propagate to caller
            box["error"] = exc

    thread = threading.Thread(
        target=runner, name="pyagent-hf-search", daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, True, None
    if "error" in box:
        return None, False, box["error"]
    return box.get("result"), False, None


def _run_dataset_search(
    *,
    query: str,
    dataset_id: str | None,
    author: str | None,
    licenses: list[str] | None,
    languages: list[str] | None,
    task_categories: list[str] | None,
    modalities: list[str] | None,
    size_categories: list[str] | None,
    extra_tags: list[str] | None,
    sort_by: str,
    sort_direction: str,
    bounded_max_results: int,
    include_samples: bool,
    sample_rows: int,
    enrichment_budget: int,
    hf_token: str | None,
) -> str:
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except Exception as exc:
        return (
            "Error: `huggingface_hub` is not installed. "
            "Install with `pip install huggingface_hub datasets` to enable "
            f"`search_hf_datasets`. Original error: {exc}"
        )

    filter_tags = _build_filter_tags(licenses, modalities, extra_tags)
    sort = _normalize_sort(sort_by)
    direction = -1 if sort_direction == "desc" else None
    token = _resolve_hf_token(hf_token)
    client = HfApi(token=token)

    list_kwargs: dict[str, Any] = {
        "sort": sort,
        "direction": direction,
        "limit": bounded_max_results,
        "full": False,
    }
    if query:
        list_kwargs["search"] = query
    if dataset_id:
        list_kwargs["dataset_name"] = dataset_id
    if author:
        list_kwargs["author"] = author
    if languages:
        list_kwargs["language"] = languages
    if task_categories:
        list_kwargs["task_categories"] = task_categories
    if size_categories:
        list_kwargs["size_categories"] = size_categories
    if filter_tags:
        list_kwargs["filter"] = filter_tags

    try:
        datasets = list(client.list_datasets(**list_kwargs))
    except Exception as exc:
        return f"Error searching datasets: {type(exc).__name__}: {exc}"

    if not datasets:
        return (
            "No datasets found matching your criteria. "
            "Try removing filters, broadening the query, or using `sort_by=\"trendingScore\"`."
        )

    reports: list[DatasetReport] = []
    enrichment_skipped = 0
    for index, ds in enumerate(datasets[:bounded_max_results]):
        report = _populate_report_from_dataset_info(ds)
        if include_samples:
            if index < enrichment_budget:
                _enrich_report_with_samples(
                    report,
                    hf_token=token,
                    sample_rows=sample_rows,
                )
            else:
                enrichment_skipped += 1
                report.sample_error = (
                    f"skipped to keep the tool fast (only the top "
                    f"{enrichment_budget} result{'s are' if enrichment_budget != 1 else ' is'} "
                    "enriched per call; raise `max_enriched` to fetch more samples)"
                )
        reports.append(report)

    output: list[str] = ["# HuggingFace Dataset Search Results", ""]
    output.append(f"**Query**: `{query or '(none)'}`")
    output.append(
        "**Filters**: "
        f"license={licenses or '-'}, language={languages or '-'}, "
        f"task={task_categories or '-'}, modality={modalities or '-'}, "
        f"size_categories={size_categories or '-'}, extra_tags={extra_tags or '-'}"
    )
    output.append(f"**Sort**: `{sort}` ({sort_direction})")
    output.append(
        f"**Returned**: {len(reports)} of top {bounded_max_results} requested")
    if include_samples:
        enriched = max(0, len(reports) - enrichment_skipped)
        output.append(
            f"**Enriched with samples**: {enriched} (skipped {enrichment_skipped})"
        )
    output.append("")
    output.append("---")
    output.append("")

    for report in reports:
        output.append(report.to_markdown(include_samples=include_samples))
        output.append("")
        output.append("---")
        output.append("")

    return "\n".join(output).rstrip() + "\n"


def search_hf_datasets(
    query: str = "",
    *,
    dataset_id: str | None = None,
    author: str | None = None,
    filter_license: Any = None,
    filter_language: Any = None,
    filter_task: Any = None,
    filter_modality: Any = None,
    filter_size_categories: Any = None,
    extra_tags: list[str] | None = None,
    sort_by: str = "downloads",
    sort_direction: str = "desc",
    max_results: int = 10,
    include_samples: bool = True,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_enriched: int = DEFAULT_MAX_ENRICHED,
    tool_timeout: float | int | None = None,
    hf_token: str | None = None,
) -> str:
    _set_default_hf_http_timeouts()

    bounded_max_results = max(1, min(int(max_results or 1), MAX_RESULTS))
    sample_rows = max(1, min(int(sample_rows or DEFAULT_SAMPLE_ROWS), 10))
    enrichment_budget = max(
        0, min(int(max_enriched or 0), bounded_max_results))
    effective_timeout = _resolve_tool_timeout(tool_timeout)

    languages = _as_list(filter_language)
    task_categories = _as_list(filter_task)
    licenses = _as_list(filter_license)
    modalities = _as_list(filter_modality)
    size_categories = _as_list(filter_size_categories)

    def work() -> str:
        return _run_dataset_search(
            query=query,
            dataset_id=dataset_id,
            author=author,
            licenses=licenses,
            languages=languages,
            task_categories=task_categories,
            modalities=modalities,
            size_categories=size_categories,
            extra_tags=extra_tags,
            sort_by=sort_by,
            sort_direction=sort_direction,
            bounded_max_results=bounded_max_results,
            include_samples=include_samples,
            sample_rows=sample_rows,
            enrichment_budget=enrichment_budget,
            hf_token=hf_token,
        )

    result, timed_out, error = _run_with_timeout(work, effective_timeout)

    if timed_out:
        return (
            f"Error: `search_hf_datasets` exceeded its {effective_timeout:.0f}s "
            "wall-clock timeout. The HF Hub or your network is likely slow or "
            "unreachable. Try narrowing filters, lowering `max_results`, "
            "lowering `max_enriched`, setting `include_samples=False`, or "
            f"raising the budget via `tool_timeout` / `{TOOL_TIMEOUT_ENV_VAR}`."
        )
    if error is not None:
        return f"Error during dataset search: {type(error).__name__}: {error}"

    return result if result is not None else "Error: dataset search returned no output."


TOOL_NAME = "search_hf_datasets"
TOOL_DESCRIPTION = (
    "Search the Hugging Face Hub for datasets and return a markdown report. "
    "Filters (license, language, task, modality, size_categories) are pushed down "
    "to the Hub when possible. The top results may be enriched with config/split "
    "metadata and a few streaming sample rows."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Free-text search query for dataset name/description.",
            "default": "",
        },
        "dataset_id": {
            "type": ["string", "null"],
            "description": "Exact dataset id to look up (e.g. 'glue').",
        },
        "author": {
            "type": ["string", "null"],
            "description": "Limit results to a specific Hub author/organization.",
        },
        "filter_license": {
            "type": ["string", "array", "null"],
            "description": "License tag(s) to require, e.g. 'mit' or ['mit','apache-2.0'].",
        },
        "filter_language": {
            "type": ["string", "array", "null"],
            "description": "Language tag(s) to require, e.g. 'en' or ['en','fr'].",
        },
        "filter_task": {
            "type": ["string", "array", "null"],
            "description": "Task category tag(s) to require, e.g. 'text-classification'.",
        },
        "filter_modality": {
            "type": ["string", "array", "null"],
            "description": "Modality tag(s) to require, e.g. 'text' or ['text','image'].",
        },
        "filter_size_categories": {
            "type": ["string", "array", "null"],
            "description": "Size category tag(s) such as '10K<n<100K'.",
        },
        "extra_tags": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Arbitrary additional Hub tag predicates to require.",
        },
        "sort_by": {
            "type": "string",
            "description": "Sort field: downloads, likes, lastModified, createdAt, trendingScore.",
            "default": "downloads",
        },
        "sort_direction": {
            "type": "string",
            "description": "'asc' or 'desc'.",
            "default": "desc",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of datasets to return (1-50).",
            "default": 10,
        },
        "include_samples": {
            "type": "boolean",
            "description": "Whether to include schema/sample previews for the top results.",
            "default": True,
        },
        "sample_rows": {
            "type": "integer",
            "description": "Sample rows to fetch from each enriched dataset (1-10).",
            "default": DEFAULT_SAMPLE_ROWS,
        },
        "max_enriched": {
            "type": "integer",
            "description": "Maximum number of top results to enrich with schema/samples.",
            "default": DEFAULT_MAX_ENRICHED,
        },
        "tool_timeout": {
            "type": ["number", "null"],
            "description": "Wall-clock timeout in seconds for the whole tool call.",
        },
    },
}
TOOL_VERSION = "1"


@click.group()
def cli() -> None:
    """PyAgent external tool: search Hugging Face Hub datasets."""


@cli.command()
def describe() -> None:
    """Print the JSON manifest used by PyAgent to register this tool."""
    click.echo(
        json.dumps(
            {
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "parameters": TOOL_PARAMETERS,
                "version": TOOL_VERSION,
            },
            ensure_ascii=False,
        )
    )


@cli.command()
@click.option(
    "--args-file",
    "args_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def invoke(args_file: Path) -> None:
    """Run the tool with arguments read from ``--args-file``."""
    try:
        arguments = json.loads(args_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        click.echo(f"Failed to read --args-file: {exc}", err=True)
        sys.exit(2)

    if not isinstance(arguments, dict):
        click.echo("--args-file must contain a JSON object.", err=True)
        sys.exit(2)

    try:
        result = search_hf_datasets(**arguments)
    except TypeError as exc:
        click.echo(f"Invalid tool arguments: {exc}", err=True)
        sys.exit(2)
    except Exception as exc:
        click.echo(f"Tool error: {exc}", err=True)
        sys.exit(1)

    click.echo(result if result is not None else "")


if __name__ == "__main__":
    cli()
