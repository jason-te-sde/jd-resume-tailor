"""Job descriptions: reading them from files and job boards, and extracting their structure."""

from tailor.jd.extract import JobDescription, extract_job
from tailor.jd.sources import JDError, JobRef, JobText, canonicalize, fetch_job, load_job
from tailor.jd.text import html_to_text

__all__ = [
    "JDError",
    "JobDescription",
    "JobRef",
    "JobText",
    "canonicalize",
    "extract_job",
    "fetch_job",
    "html_to_text",
    "load_job",
]
