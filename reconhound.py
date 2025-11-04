"""ReconHound web scraper for gathering reconnaissance data for penetration testers.

This module provides a resilient web scraping utility that focuses on ethically collecting
publicly available information such as employee emails, names, and job titles from a list
of domains. The scraper is careful to respect the websites' ``robots.txt`` directives and
includes robust error handling so that unexpected failures do not crash the process.

The script can be executed directly and supports exporting results either as JSON or XML.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

# Configure a default logger for the module.
logger = logging.getLogger("reconhound")

# A simple and conservative pattern for capturing email addresses.
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Common job title keywords to help flag potential employee-related content.
JOB_TITLE_KEYWORDS = {
    "engineer",
    "developer",
    "manager",
    "director",
    "cto",
    "ceo",
    "founder",
    "security",
    "analyst",
    "consultant",
    "lead",
    "officer",
    "specialist",
    "administrator",
}

WORD_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z'-]*)\b")
PHONE_REGEX = re.compile(r"\+?\d[\d\s().-]{6,}\d")
FIRST_NAMES_PATH = Path(__file__).resolve().with_name("first_names.txt")


@lru_cache(maxsize=1)
def load_first_names() -> Set[str]:
    """Load first names from the bundled wordlist."""
    first_names: Set[str] = set()
    try:
        with FIRST_NAMES_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                first_names.add(name.lower())
    except OSError as exc:
        logger.debug("First-name wordlist %s unavailable: %s", FIRST_NAMES_PATH, exc)
    return first_names


@dataclass
class Finding:
    """Container for a single piece of scraped reconnaissance data."""

    type: str
    value: str
    context: Optional[str]
    source_url: str

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Return a serializable dictionary representation of the finding."""
        return asdict(self)


class RobotsCache:
    """Cache ``robots.txt`` parsers to minimise repeated network calls."""

    def __init__(self, session: requests.Session):
        self._session = session
        self._cache: Dict[str, robotparser.RobotFileParser] = {}

    def can_fetch(self, url: str, user_agent: str) -> bool:
        """Determine whether ``user_agent`` is allowed to fetch ``url``."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        if robots_url not in self._cache:
            parser = robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                response = self._session.get(robots_url, timeout=5)
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    parser.parse([])
            except requests.RequestException:
                # When robots cannot be retrieved we fall back to denying the request
                # to err on the side of caution.
                logger.warning("Unable to retrieve robots.txt from %s", robots_url)
                parser.disallow_all = True
            self._cache[robots_url] = parser

        parser = self._cache[robots_url]
        return parser.can_fetch(user_agent, url)


class ReconHoundScraper:
    """High-level scraper class that orchestrates data collection for domains."""

    def __init__(
        self,
        user_agent: str = "ReconHound/1.0 (+https://example.com/reconhound)",
        respect_robots_txt: bool = True,
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.user_agent = user_agent
        self.robots_cache = RobotsCache(self.session)
        self.respect_robots_txt = respect_robots_txt

    def scrape_domains(self, domains: Iterable[str], max_pages_per_domain: int = 5) -> Dict[str, List[Finding]]:
        """Scrape a series of domains, returning findings grouped by domain.

        Parameters
        ----------
        domains:
            An iterable of domain names (e.g., ``example.com``).
        max_pages_per_domain:
            Safety limit that avoids crawling too aggressively.
        """
        aggregated: Dict[str, List[Finding]] = {}
        for domain in domains:
            findings: List[Finding] = []
            visited: Set[str] = set()
            queue: List[str] = []
            normalized_domain = domain.strip()
            seen_values: Dict[str, Set[str]] = defaultdict(set)

            if not normalized_domain:
                aggregated[domain] = findings
                continue

            candidates = self._build_candidate_start_urls(normalized_domain)

            start_url: Optional[str] = None
            allowed_path_prefix = "/"

            for candidate_url, candidate_prefix in candidates:
                if self._allowed(candidate_url) and self._fetchable(candidate_url):
                    start_url = candidate_url
                    allowed_path_prefix = candidate_prefix
                    break

            if not start_url:
                logger.warning("Skipping %s: no accessible HTTP(S) endpoint", domain)
                aggregated[domain] = findings
                continue

            queue.append(start_url)

            while queue and len(visited) < max_pages_per_domain:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                page_findings, links = self._scrape_page(url, allowed_path_prefix)
                for finding in page_findings:
                    value_key = finding.value.lower()
                    type_seen = seen_values[finding.type]
                    if value_key in type_seen:
                        continue
                    type_seen.add(value_key)
                    findings.append(finding)

                for link in links:
                    if len(visited) + len(queue) >= max_pages_per_domain:
                        break
                    if link not in visited and link not in queue and self._allowed(link):
                        queue.append(link)

            aggregated[domain] = findings
        return aggregated

    def _build_candidate_start_urls(self, normalized_domain: str) -> List[Tuple[str, str]]:
        """Return possible start URLs paired with their allowed path prefixes."""
        candidates: List[Tuple[str, str]] = []
        parsed = urlparse(normalized_domain)

        if parsed.scheme and parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
            scheme = parsed.scheme.lower()
            normalized_path = self._normalize_allowed_path(parsed.path or "/")
            rebuilt = urlunparse((scheme, parsed.netloc, normalized_path, "", "", ""))
            candidates.append((rebuilt, normalized_path))
            return candidates

        trimmed = normalized_domain.strip("/")
        if not trimmed:
            return candidates

        if "/" in trimmed:
            host, remainder = trimmed.split("/", 1)
            path = f"/{remainder}"
        else:
            host = trimmed
            path = "/"

        if not host:
            return candidates

        normalized_path = self._normalize_allowed_path(path)
        for scheme in ("https", "http"):
            rebuilt = f"{scheme}://{host}{normalized_path}"
            candidates.append((rebuilt, normalized_path))

        return candidates

    def _fetchable(self, url: str) -> bool:
        """Check if the URL responds successfully with a GET request."""
        try:
            response = self.session.head(url, allow_redirects=True, timeout=5)
            if response.status_code < 400:
                return True
            if response.status_code in {405, 501}:
                # Some servers do not support HEAD; allow a follow-up GET attempt.
                return True
        except requests.RequestException as exc:
            logger.info("HEAD request failed for %s: %s", url, exc)
            return True
        return False

    def _allowed(self, url: str) -> bool:
        """Return True if the URL is allowed according to robots.txt."""
        if not self.respect_robots_txt:
            return True
        allowed = self.robots_cache.can_fetch(url, self.user_agent)
        if not allowed:
            logger.info("Blocked by robots.txt: %s", url)
        return allowed

    def _scrape_page(self, url: str, allowed_path_prefix: str) -> (List[Finding], List[str]):
        """Scrape a single page, returning findings and discovered in-scope internal links."""
        findings: List[Finding] = []
        links: List[str] = []

        try:
            response = self.session.get(url, timeout=(5, 10))
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return findings, links

        soup = BeautifulSoup(response.text, "html.parser")
        base_url = response.url
        base_path = urlparse(base_url).path

        if not self._is_within_allowed_path(base_path, allowed_path_prefix):
            logger.debug(
                "Skipping %s redirected to %s outside allowed path %s",
                url,
                base_url,
                allowed_path_prefix,
            )
            return findings, links

        findings.extend(self._extract_emails(soup, base_url))
        findings.extend(self._extract_phone_numbers(soup, base_url))
        findings.extend(self._extract_people_info(soup, base_url))

        links.extend(self._extract_internal_links(soup, base_url, allowed_path_prefix))
        return findings, links

    def _extract_emails(self, soup: BeautifulSoup, url: str) -> List[Finding]:
        """Extract email addresses from the soup and return them as findings."""
        findings: List[Finding] = []
        seen_emails: Set[str] = set()

        # Email addresses surfaced via ``mailto:`` links.
        for mail_link in soup.select('a[href^="mailto:"]'):
            email = mail_link.get("href", "").split(":", 1)[-1]
            if email and EMAIL_REGEX.fullmatch(email):
                normalized = email.lower()
                if normalized in seen_emails:
                    continue
                seen_emails.add(normalized)
                findings.append(Finding("email", email, mail_link.get_text(strip=True) or None, url))

        # Fallback: parse raw text content for email-like strings.
        for match in EMAIL_REGEX.findall(soup.get_text("\n")):
            normalized = match.lower()
            if normalized in seen_emails:
                continue
            seen_emails.add(normalized)
            findings.append(Finding("email", match, None, url))

        return findings

    def _extract_phone_numbers(self, soup: BeautifulSoup, url: str) -> List[Finding]:
        """Extract phone numbers from the soup."""
        findings: List[Finding] = []
        seen_numbers: Set[str] = set()

        for tel_link in soup.select('a[href^="tel:"]'):
            raw_number = tel_link.get("href", "").split(":", 1)[-1]
            normalized = self._normalize_phone_value(raw_number)
            if not normalized or normalized in seen_numbers:
                continue
            seen_numbers.add(normalized)
            findings.append(Finding("phone", normalized, tel_link.get_text(strip=True) or raw_number, url))

        text_content = soup.get_text(" ")
        for match in PHONE_REGEX.finditer(text_content):
            raw_candidate = match.group().strip()
            normalized = self._normalize_phone_value(raw_candidate)
            if not normalized or normalized in seen_numbers:
                continue
            seen_numbers.add(normalized)
            findings.append(Finding("phone", normalized, raw_candidate, url))

        return findings

    def _extract_people_info(self, soup: BeautifulSoup, url: str) -> List[Finding]:
        """Attempt to extract names and job titles from page content."""
        findings: List[Finding] = []
        text_blocks = [t.strip() for t in soup.stripped_strings if len(t.strip()) <= 120]
        first_names = load_first_names()
        seen_identifiers: Set[str] = set()

        for block in text_blocks:
            lower_block = block.lower()
            used_word_indices: Set[int] = set()

            # Identify likely names using the first-name wordlist and capitalized surnames.
            words = [match.group(0) for match in WORD_PATTERN.finditer(block)]
            for idx, raw_first in enumerate(words):
                if idx in used_word_indices:
                    continue
                if not raw_first or not raw_first[0].isupper():
                    continue
                if len(raw_first) <= 1 or not any(ch.islower() for ch in raw_first[1:]):
                    continue

                normalized_first = raw_first.lower()
                allowed_first = normalized_first in first_names
                if not allowed_first and idx == 0 and len(words) == 2:
                    raw_last_candidate = words[1]
                    if raw_last_candidate and raw_last_candidate[0].isupper() and any(
                        ch.islower() for ch in raw_last_candidate[1:]
                    ):
                        allowed_first = True
                if not allowed_first:
                    continue
                first_key = f"first:{normalized_first}"

                raw_last: Optional[str] = None
                last_idx: Optional[int] = None
                for candidate_idx in range(idx + 1, len(words)):
                    candidate = words[candidate_idx]
                    if not candidate or not candidate[0].isupper():
                        break
                    if len(candidate) <= 1 or not any(ch.islower() for ch in candidate[1:]):
                        continue
                    raw_last = candidate
                    last_idx = candidate_idx
                    break

                if raw_last:
                    normalized_last = raw_last.lower()
                    full_key = f"full:{normalized_first}:{normalized_last}"
                    if full_key in seen_identifiers:
                        continue
                    seen_identifiers.add(full_key)
                    seen_identifiers.add(first_key)
                    if last_idx is not None:
                        # mark words used in this full name to avoid treating surname as standalone
                        used_word_indices.update({idx, last_idx})
                    full_name = f"{raw_first} {raw_last}"
                    findings.append(Finding("name", full_name, block, url))
                else:
                    if first_key in seen_identifiers:
                        continue
                    seen_identifiers.add(first_key)
                    findings.append(Finding("name", raw_first, block, url))

            # Identify potential job titles.
            if any(keyword in lower_block for keyword in JOB_TITLE_KEYWORDS):
                findings.append(Finding("occupation", block, block, url))

        return findings

    @staticmethod
    def _normalize_phone_value(raw: str) -> Optional[str]:
        """Return a canonical phone representation or None if invalid."""
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 7:
            return None
        if raw.strip().startswith("+"):
            return f"+{digits}"
        return digits

    @staticmethod
    def _normalize_allowed_path(path: str) -> str:
        if not path:
            return "/"
        if not path.startswith("/"):
            path = "/" + path
        if path == "/":
            return "/"
        # Treat paths ending with a slash as directories.
        if path.endswith("/"):
            return path.rstrip("/") + "/"
        last_segment = path.rsplit("/", 1)[-1]
        # If the last segment looks like a file (contains a dot), keep it as-is.
        if "." in last_segment:
            return path
        return path + "/"

    @staticmethod
    def _is_within_allowed_path(candidate_path: str, allowed_prefix: str) -> bool:
        if allowed_prefix == "/":
            return True
        candidate = candidate_path or "/"
        if not candidate.startswith("/"):
            candidate = "/" + candidate
        if allowed_prefix.endswith("/"):
            if candidate == allowed_prefix[:-1]:
                return True
            return candidate.startswith(allowed_prefix)
        # File-scoped crawl: require an exact match.
        return candidate == allowed_prefix

    def _extract_internal_links(
        self,
        soup: BeautifulSoup,
        base_url: str,
        allowed_path_prefix: str,
    ) -> List[str]:
        """Extract internal links within the allowed domain/path scope."""
        links: List[str] = []
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if href.startswith("#"):
                continue
            joined = urljoin(base_url, href)
            parsed_link = urlparse(joined)
            if parsed_link.netloc == base_domain and parsed_link.scheme in {"http", "https"}:
                if self._is_within_allowed_path(parsed_link.path, allowed_path_prefix):
                    links.append(joined)

        return links


def output_data(data: Dict[str, List[Finding]], format_type: str, output_file: Optional[str] = None) -> str:
    """Serialize ``data`` into JSON or XML and optionally persist it to ``output_file``.

    Returns the serialized string for convenience.
    """
    format_type = format_type.lower()
    serialized: str

    if format_type == "json":
        payload = {domain: [finding.to_dict() for finding in findings] for domain, findings in data.items()}
        serialized = json.dumps(payload, indent=2)
    elif format_type == "xml":
        from xml.etree.ElementTree import Element, SubElement, tostring

        root = Element("reconhound")
        for domain, findings in data.items():
            domain_node = SubElement(root, "domain", name=domain)
            for finding in findings:
                finding_node = SubElement(domain_node, "finding", type=finding.type)
                SubElement(finding_node, "value").text = finding.value
                if finding.context:
                    SubElement(finding_node, "context").text = finding.context
                SubElement(finding_node, "source").text = finding.source_url
        serialized = tostring(root, encoding="unicode")
    else:
        raise ValueError("format_type must be either 'json' or 'xml'")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write(serialized)
    return serialized


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ReconHound - ethical reconnaissance web scraper")
    parser.add_argument(
        "domains",
        nargs="*",
        help="Domain names to scrape (e.g. example.com).",
    )
    parser.add_argument(
        "--input-file",
        help="Optional path to a file containing domains (one per line).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "xml"],
        default="json",
        help="Output serialization format.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save serialized output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum number of pages to visit per domain.",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Override robots.txt and crawl even when disallowed.",
    )
    return parser.parse_args(argv)


def load_domains(args: argparse.Namespace) -> List[str]:
    """Merge domains provided via CLI arguments and an optional input file."""
    domains = list(args.domains)
    if args.input_file:
        try:
            with open(args.input_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        domains.append(line)
        except OSError as exc:
            logger.error("Failed to read domain file %s: %s", args.input_file, exc)
    return domains


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for command-line execution."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    domains = load_domains(args)
    if not domains:
        logger.error("No domains provided. Use --help for usage information.")
        return 1

    scraper = ReconHoundScraper(respect_robots_txt=not args.ignore_robots)
    findings = scraper.scrape_domains(domains, max_pages_per_domain=args.max_pages)

    serialized = output_data(findings, args.format, output_file=args.output)
    print(serialized)

    return 0


if __name__ == "__main__":
    sys.exit(main())
