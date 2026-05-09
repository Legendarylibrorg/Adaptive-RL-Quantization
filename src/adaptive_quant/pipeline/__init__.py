"""Research pipeline building blocks (VCS stamps, analysis, reporting, benchmark warnings)."""

from adaptive_quant.pipeline.analysis_runner import run_research_analysis
from adaptive_quant.pipeline.benchmark_warn import warn_if_benchmarks_are_large
from adaptive_quant.pipeline.report_markdown import write_research_report_markdown
from adaptive_quant.pipeline.vcs import git_commit_hash

__all__ = [
    "git_commit_hash",
    "run_research_analysis",
    "warn_if_benchmarks_are_large",
    "write_research_report_markdown",
]
