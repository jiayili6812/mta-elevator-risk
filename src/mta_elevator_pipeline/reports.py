"""Formal JSON and Markdown report writers."""

from __future__ import annotations

import json
from pathlib import Path


def write_json_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_audit_markdown(report: dict[str, object], path: Path) -> None:
    source = report["source_validation"]
    eligibility = report["eligibility"]
    age = report["missing_age"]
    frozen = report["frozen_test_count_confirmation"]
    lines = [
        "# Standalone Pipeline Real-Data Audit",
        "",
        "## Result",
        "",
        f"- Validation status: **{report['status']}**",
        f"- Source rows: {source['rows']}",
        f"- Equipment: {source['equipment']}",
        f"- Date range: {source['start_month']} through {source['end_month']}",
        f"- Nonconsecutive transitions: {source['nonconsecutive_transitions']}",
        "",
        "## Eligibility",
        "",
        f"- X-suffixed rows retained in audit: {eligibility['x_suffix_rows']}",
        f"- X-suffixed equipment excluded from modeling: {eligibility['x_suffix_equipment']}",
        f"- Eligible non-X equipment: {eligibility['eligible_equipment']}",
        f"- Newly introduced eligible equipment: {eligibility['newly_introduced_equipment']}",
        "",
        "## Feature History",
        "",
    ]
    for name, details in report["feature_set_history"].items():
        lines.append(
            f"- `{name}`: requires {details['minimum_consecutive_months']} consecutive "
            f"month(s); {details['eligible_rows']} rows and "
            f"{details['eligible_equipment']} equipment qualify."
        )
    lines.extend(
        [
            "",
            "## Missing Age",
            "",
            f"- Missing non-X rows: {age['missing_non_x_rows']}",
            f"- Affected equipment: {age['affected_equipment']}",
            f"- All occur on first observed record: {age['all_on_first_observed_record']}",
            f"- Strategy: {age['strategy']}",
            "",
            "## Rolling Validation",
            "",
        ]
    )
    for fold in report["rolling_validation_folds"]:
        lines.append(
            f"- `{fold['name']}`: train rows {fold['train_rows']}; "
            f"validation rows {fold['validation_rows']}."
        )
    lines.extend(
        [
            "",
            "## Frozen-Test Guardrail",
            "",
            f"- Target-eligible frozen rows before feature-history filtering: "
            f"{frozen['target_eligible_rows_before_feature_history_filter']}",
            f"- Full-feature frozen rows: {frozen['rows']}",
            f"- Counts match locked expectations: {frozen['matches_locked_expectations']}",
            f"- Frozen labels exposed by audit command: {frozen['labels_exposed']}",
            "",
            "## Target Reports",
            "",
        ]
    )
    for name, target in report["development_target_prevalence"].items():
        lines.append(
            f"- `{name}` development-only prevalence: {target['prevalence']:.6f} "
            f"({target['positive_rows']} / {target['labeled_rows']})."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
