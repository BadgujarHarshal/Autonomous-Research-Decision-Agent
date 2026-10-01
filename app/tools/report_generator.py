# app/tools/report_generator.py

from __future__ import annotations

from typing import Any, Dict, List

from app.tools.base import BaseTool


class ReportGeneratorTool(BaseTool):
    name = "generate_report"
    description = (
        "Generates a structured research report from a title, summary, "
        "findings, and optional recommendations."
    )

    @property
    def argument_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                },
                "summary": {
                    "type": "string",
                },
                "findings": {
                    "type": "array",
                },
                "recommendations": {
                    "type": "array",
                },
            },
            "required": ["title", "summary", "findings"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        self.validate_arguments(arguments)

        title = arguments.get("title")
        summary = arguments.get("summary")
        findings = arguments.get("findings")
        recommendations = arguments.get(
            "recommendations",
            [],
        )

        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string.")

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be a non-empty string.")

        if not isinstance(findings, list):
            raise ValueError("findings must be a list.")

        if not isinstance(recommendations, list):
            raise ValueError("recommendations must be a list.")

        lines = [
            f"# {title.strip()}",
            "",
            "## Executive Summary",
            "",
            summary.strip(),
            "",
            "## Findings",
            "",
        ]

        if findings:
            for finding in findings:
                lines.append(f"- {str(finding).strip()}")
        else:
            lines.append("- No findings provided.")

        if recommendations:
            lines.extend(
                [
                    "",
                    "## Recommendations",
                    "",
                ]
            )

            for recommendation in recommendations:
                lines.append(
                    f"- {str(recommendation).strip()}"
                )

        report = "\n".join(lines)

        return {
            "title": title.strip(),
            "report": report,
            "finding_count": len(findings),
            "recommendation_count": len(recommendations),
        }