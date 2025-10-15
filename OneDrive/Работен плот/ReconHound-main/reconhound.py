"""ReconHound web scraper for gathering reconnaissance data for penetration testers.

This module provides a resilient web scraping utility that focuses on ethically collecting
publicly available information such as employee emails, names, and job titles from a list
of domains. The scraper is careful to respect the websites' ``robots.txt`` directives and
includes robust error handling so that unexpected failures do not crash the process.

The script can be executed directly and supports exporting results either as JSON or XML.
"""
# Test from Stoyan

# Second test from Stoyan

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import urljoin, urlparse
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

# A loose pattern for detecting likely first + last name combinations.
NAME_REGEX = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b")


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

    def __init__(self, user_agent: str = "ReconHound/1.0 (+https://example.com/reconhound)"):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.user_agent = user_agent
        self.robots_cache = RobotsCache(self.session)

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

            for scheme in ("https://", "http://"):
                start_url = f"{scheme}{domain.strip('/')}/"
                if self._allowed(start_url) and self._fetchable(start_url):
                    queue.append(start_url)
                    break
            else:
                logger.warning("Skipping %s: no accessible HTTP(S) endpoint", domain)
                aggregated[domain] = findings
                continue

            while queue and len(visited) < max_pages_per_domain:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                page_findings, links = self._scrape_page(url)
                findings.extend(page_findings)

                for link in links:
                    if len(visited) + len(queue) >= max_pages_per_domain:
                        break
                    if link not in visited and link not in queue and self._allowed(link):
                        queue.append(link)

            aggregated[domain] = findings
        return aggregated

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
        allowed = self.robots_cache.can_fetch(url, self.user_agent)
        if not allowed:
            logger.info("Blocked by robots.txt: %s", url)
        return allowed

    def _scrape_page(self, url: str) -> (List[Finding], List[str]):
        """Scrape a single page, returning findings and discovered internal links."""
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

        findings.extend(self._extract_emails(soup, base_url))
        findings.extend(self._extract_people_info(soup, base_url))

        links.extend(self._extract_internal_links(soup, base_url))
        return findings, links

    def _extract_emails(self, soup: BeautifulSoup, url: str) -> List[Finding]:
        """Extract email addresses from the soup and return them as findings."""
        findings: List[Finding] = []

        # Email addresses surfaced via ``mailto:`` links.
        for mail_link in soup.select('a[href^="mailto:"]'):
            email = mail_link.get("href", "").split(":", 1)[-1]
            if email and EMAIL_REGEX.fullmatch(email):
                findings.append(Finding("email", email, mail_link.get_text(strip=True) or None, url))

        # Fallback: parse raw text content for email-like strings.
        for match in EMAIL_REGEX.findall(soup.get_text("\n")):
            findings.append(Finding("email", match, None, url))

        return findings

    def _extract_people_info(self, soup: BeautifulSoup, url: str) -> List[Finding]:
        """Attempt to extract names and job titles from page content."""
        findings: List[Finding] = []
        text_blocks = [t.strip() for t in soup.stripped_strings if len(t.strip()) <= 120]

        for block in text_blocks:
            lower_block = block.lower()

            # Identify likely names.
            for match in NAME_REGEX.findall(block):
                findings.append(Finding("name", match, block, url))

            # Identify potential job titles.
            if any(keyword in lower_block for keyword in JOB_TITLE_KEYWORDS):
                findings.append(Finding("occupation", block, block, url))

        return findings

    def _extract_internal_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract internal links within the same domain as ``base_url``."""
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

    scraper = ReconHoundScraper()
    findings = scraper.scrape_domains(domains, max_pages_per_domain=args.max_pages)

    serialized = output_data(findings, args.format, output_file=args.output)
    print(serialized)

    return 0


if __name__ == "__main__":
    sys.exit(main())
