#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Data preparation module."""

import hashlib
from pathlib import Path
from xml.etree import ElementTree

import httpx
import orjson
import pydash as py_
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

#
# Constants
#
HTTP_TIMEOUT = 30.0
"""Per-request timeout, in seconds."""

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
"""XML namespace used by ``sitemap.xml`` documents."""

CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]
"""Ordered separators used by the recursive chunker."""


#
# Internal functions - chunking
#
def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
	"""Prepend the tail of each chunk to the next one.

	Args:
		chunks (list[str]): The ordered list of chunks.

		chunk_overlap (int): Number of trailing characters to carry over.

	Returns:
		list[str]: The list of chunks with overlap applied.
	"""
	# prepend the tail of each chunk to the next one
	result = [chunks[0]]

	for i in range(1, len(chunks)):
		# get the previous chunk "tail"
		prev_tail = chunks[i - 1][-chunk_overlap:]

		# prepend the previous chunk "tail" to the current
		# chunk and strip whitespace
		result.append((prev_tail + " " + chunks[i]).strip())

	# return!
	return result


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
	"""Split text at fixed character boundaries when no separator applies.

	Args:
		text (str): The text to split.

		chunk_size (int): The maximum size of each chunk.

		chunk_overlap (int): The overlap between consecutive chunks.

	Returns:
		list[str]: The list of fixed-size chunks.
	"""
	start = 0
	chunks = []

	# iterate over the text
	while start < len(text):
		# get the end index
		end = start + chunk_size

		# add the chunk to the list
		chunks.append(text[start:end])

		# update the start index
		start = end - chunk_overlap if chunk_overlap > 0 else end

	# return
	return chunks


def _split_with_separator(
	text: str, sep: str, chunk_size: int, chunk_overlap: int, separators: list[str]
) -> list[str]:
	"""Split on a separator and merge pieces into chunks of the target size.

	Args:
		text (str): The text to split.

		sep (str): The separator to split on.

		chunk_size (int): The maximum size of each chunk.

		chunk_overlap (int): The overlap between consecutive chunks.

		separators (list[str]): The remaining separator ladder for recursion.

	Returns:
		list[str]: The list of chunks.
	"""
	pieces = text.split(sep)

	# re-attach the period on sentence splits
	if sep == ". ":
		pieces = [p + "." if i < len(pieces) - 1 else p for i, p in enumerate(pieces)]

	current = ""
	chunks = []
	for piece in pieces:
		# get the candidate chunk
		candidate = (current + sep + piece).strip() if current else piece.strip()

		# if the candidate chunk is smaller than the target
		# size, add it to the list
		if len(candidate) <= chunk_size:
			current = candidate
			continue

		# add the current chunk to the list
		if current:
			chunks.append(current)

		# if the piece is larger than the target size, split it with
		# finer separators
		if len(piece.strip()) > chunk_size:
			# get the remaining separators
			remaining = separators[separators.index(sep) + 1 :]

			# split the piece with the remaining separators
			chunks.extend(
				_recursive_split(piece.strip(), remaining, chunk_size, chunk_overlap)
			)

			# reset the current chunk
			current = ""

		else:
			# reset the current chunk
			current = piece.strip()

	# add the current chunk to the list
	if current:
		chunks.append(current)

	# apply overlap in chunks if required
	if chunk_overlap > 0 and len(chunks) > 1:
		chunks = _apply_overlap(chunks, chunk_overlap)

	# return chunks
	return chunks


def _recursive_split(
	text: str, separators: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
	"""Split text using the first separator that applies.

	Args:
		text (str): The text to split.

		separators (list[str]): The ordered separator ladder.

		chunk_size (int): The maximum size of each chunk.

		chunk_overlap (int): The overlap between consecutive chunks.

	Returns:
		list[str]: The list of chunks.
	"""
	if len(text) <= chunk_size:
		return [text]

	for sep in separators:
		if sep in text:
			return _split_with_separator(
				text, sep, chunk_size, chunk_overlap, separators
			)

	return _hard_split(text, chunk_size, chunk_overlap)


#
# Internal functions - identifiers and HTTP
#
def _chunk_id(prefix: str, *parts: str) -> str:
	"""Create chunk identifier.

	Args:
		prefix (str): Source prefix (e.g. ``"book"``).

		parts (str): Values hashed together to form the identifier.

	Returns:
		str: Identifier of the form ``"<prefix>-<16 hex chars>"``.
	"""
	digest = hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]

	return f"{prefix}-{digest}"


def _fetch_sitemap_urls(client: httpx.Client, base_url: str) -> list[str]:
	"""Fetch ``sitemap.xml`` of the SITS documentation.

	Args:
		client (httpx.Client): An open HTTP client.

		base_url (str): The base URL of the site.

	Returns:
		list[str]: The list of page URLs declared in the sitemap.
	"""
	# request
	response = client.get(f"{base_url}/sitemap.xml")
	response.raise_for_status()

	# parse
	root = ElementTree.fromstring(response.text)
	locs = root.findall(".//sm:loc", SITEMAP_NS)

	# return
	return py_.compact(py_.map_(locs, lambda loc: loc.text))


#
# Internal functions - SITS book (Quarto)
#
def _extract_chapter_title(soup: BeautifulSoup) -> str | None:
	"""Extract the page/chapter title.

	Extract the page/chapter title from an ``<h1>`` or ``<title>`` tag.

	Args:
		soup (BeautifulSoup): The parsed page.

	Returns:
		str | None: The chapter title, or ``None`` if absent.
	"""
	heading = soup.find("h1") or soup.find("title")

	# return
	return heading.get_text(strip=True) if heading else None


def _split_into_sections(main: Tag) -> list[tuple[str | None, str]]:
	"""Split content into groups.

	Split the main content into ``(title, text)`` groups on ``<h2>`` and ``<h3>``.

	Args:
		main (Tag): The page's main content.

	Returns:
		list[tuple[str | None, str]]: A list of ``(section_title, section_text)`` tuples.
	"""
	sections = []

	current_title = None
	current_texts = []

	# iterate over the children of the main content
	for element in main.children:
		# skip if not a tag
		if not isinstance(element, Tag):
			continue

		# if the element is a heading, add the current text to the sections
		if element.name in ("h2", "h3"):
			# if content has been collected, add it to the sections
			if current_texts:
				sections.append((current_title, "\n\n".join(current_texts)))

			# update current title variable
			current_title = element.get_text(strip=True)

			# reset current texts
			current_texts = []

		else:
			text = element.get_text(separator=" ", strip=True)

			if text:
				current_texts.append(text)

	# add last section
	if current_texts:
		sections.append((current_title, "\n\n".join(current_texts)))

	return sections


def _parse_book_page(
	url: str, html: str, chunk_size: int, chunk_overlap: int
) -> list[dict]:
	"""Extract and chunk the main content of a SITS book page.

	Args:
		url (str): The page URL.

		html (str): The raw HTML.

		chunk_size (int): The maximum size of each chunk.

		chunk_overlap (int): The overlap between consecutive chunks.

	Returns:
		list[dict]: The list of chunk dictionaries for the page.
	"""
	# parse
	soup = BeautifulSoup(html, "lxml")
	main = soup.find("main") or soup.find("div", {"class": "content"})

	# skip if not a tag
	if not isinstance(main, Tag):
		return []

	# extract chapter title
	chapter = _extract_chapter_title(soup)

	# create prefix
	prefix = "\n".join(py_.compact([chapter and f"Chapter: {chapter}", f"URL: {url}"]))

	chunks = []

	# iterate over the sections
	for section_title, section_text in _split_into_sections(main):
		# skip if the section text is empty
		if not section_text.strip():
			continue

		# iterate over the chunks of the section text
		for i, text in enumerate(chunk_text(section_text, chunk_size, chunk_overlap)):
			# create chunk dictionary
			chunks.append(
				{
					"id": _chunk_id("book", url, section_title or "", str(i)),
					"text": f"{prefix}\n\n{text}",
					"source": "book",
					"title": section_title or chapter,
					"url": url,
					"function_name": None,
				}
			)

	# return!
	return chunks


#
# Internal functions - SITS function reference (pkgdown)
#
def _function_name_from_url(url: str) -> str:
	"""Extract the canonical function name from a pkgdown reference URL.

	Args:
		url (str): The reference page URL.

	Returns:
		str: The function name (URL slug without the ``.html`` suffix).
	"""
	return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")


def _extract_reference_sections(main: Tag) -> list[tuple[str, str]]:
	"""Extract named sections.

	Args:
		main (Tag): The reference page's main content element.

	Returns:
		list[tuple[str, str]]: A list of ``(section_name, section_text)`` tuples,
								starting with the description.
	"""
	sections = []

	# iterate over the headings
	for heading in main.find_all(["h2", "h3"]):
		# get the section name
		section_name = heading.get_text(strip=True)

		# iterate over the siblings
		content_parts = []

		# iterate over the siblings
		for sibling in heading.find_next_siblings():
			if not isinstance(sibling, Tag):
				continue

			# skip if the sibling is a heading
			if sibling.name in ("h2", "h3"):
				break

			# get the text of the sibling
			text = sibling.get_text(separator=" ", strip=True)

			# add the text to the content parts
			if text:
				content_parts.append(text)

		# add the section to the sections
		if content_parts:
			sections.append((section_name, "\n".join(content_parts)))

	# extract the description before the first heading
	desc_parts = []
	for child in main.children:
		# skip if not a tag
		if not isinstance(child, Tag):
			continue

		# skip if the child is a heading
		if child.name in ("h1", "h2", "h3"):
			# break the loop
			break

		# get the text of the child
		text = child.get_text(separator=" ", strip=True)

		# add the text to the description parts
		if text:
			desc_parts.append(text)

	# add the description to the sections
	if desc_parts:
		sections.insert(0, ("Description", "\n".join(desc_parts)))

	# return!
	return sections


def _parse_reference_page(url: str, html: str) -> dict | None:
	"""Build a single full-text function record.

	Args:
		url (str): The reference page URL.

		html (str): The raw HTML.

	Returns:
		dict | None: function record, or ``None`` if the page has no function content.
	"""
	soup = BeautifulSoup(html, "lxml")
	main = soup.find("main") or soup.find("div", {"role": "main"})

	# skip if not a tag
	if not isinstance(main, Tag):
		return None

	# extract the function name
	function_name = _function_name_from_url(url)

	# extract the title
	heading = soup.find("h1")
	title = heading.get_text(strip=True) if heading else function_name

	# extract the sections
	sections = _extract_reference_sections(main)

	# join the sections into the body
	body = "\n\n".join(
		f"{name}:\n{content}" for name, content in sections if content.strip()
	)

	# skip if the body is empty
	if not body:
		return None

	# create the text
	text = f"Function: {function_name}\nTitle: {title}\nDocs: {url}\n\n{body}"

	# create the record
	return {
		"id": _chunk_id("function", function_name),
		"text": text,
		"source": "function",
		"title": title,
		"url": url,
		"function_name": function_name,
	}


#
# Internal functions - reference articles
#
def _render_reference(entry: dict) -> str:
	"""Render a reference entry as a citation block.

	Args:
		entry (dict): A reference entry from ``references.json``.

	Returns:
		str: The rendered citation block.
	"""
	authors = ", ".join(py_.get(entry, "authors", []))
	domains = ", ".join(py_.get(entry, "domain", []))

	# build the header lines, dropping the ones without a value
	header = py_.compact(
		[
			f"Title: {py_.get(entry, 'title', 'Untitled')}",
			authors and f"Authors: {authors}",
			f"Year: {py_.get(entry, 'year')} | Venue: {py_.get(entry, 'venue')}",
			domains and f"Topics: {domains}",
			# absent for entries published without a DOI
			py_.get(entry, "doi") and f"DOI: {py_.get(entry, 'doi')}",
			f"URL: {py_.get(entry, 'url')}",
		]
	)

	# get the abstract
	abstract = py_.get(entry, "abstract", "").strip()

	# return the header on its own when there is no abstract
	return "\n".join(header) + (f"\n\n{abstract}" if abstract else "")


#
# Public functions
#
def chunk_text(
	text: str,
	chunk_size: int = 1000,
	chunk_overlap: int = 200,
) -> list[str]:
	"""Split text into overlapping chunks.

	Splits on paragraph breaks first, then newlines, sentences and spaces.
	Each chunk is at most ``chunk_size`` characters with ``chunk_overlap``
	characters of overlap between neighbours.

	Args:
		text (str): The text to split.

		chunk_size (int): The maximum size of each chunk.

		chunk_overlap (int): The overlap between consecutive chunks.

	Returns:
		list[str]: The list of chunks.
	"""
	if not text or not text.strip():
		return []

	return _recursive_split(
		text=text.strip(),
		separators=CHUNK_SEPARATORS,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)


def parse_documentation(
	base_url: str,
	chunk_size: int = 1000,
	chunk_overlap: int = 200,
) -> list[dict]:
	"""Fetch and chunk every page of the SITS book.

	Args:
		base_url (str): The base URL of the SITS book.

		chunk_size (int): The maximum size of each chunk.

		chunk_overlap (int): The overlap between consecutive chunks.

	Returns:
		list[dict]: The list of book chunk dictionaries.

	Raises:
		httpx.HTTPError: If the sitemap cannot be fetched.
	"""
	chunks = []

	# define client
	with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
		# fetch the sitemap URLs
		urls = _fetch_sitemap_urls(client, base_url)

		# iterate over the URLs
		for url in tqdm(urls, desc="Parsing SITS book pages", unit="page"):
			# fetch the page
			try:
				response = client.get(url)
				response.raise_for_status()

			except httpx.HTTPError as error:
				tqdm.write(f"Skipping {url}: {error}")
				continue

			# parse the page
			chunks.extend(
				_parse_book_page(
					url=url,
					html=response.text,
					chunk_size=chunk_size,
					chunk_overlap=chunk_overlap,
				)
			)

	# return!
	return chunks


def parse_reference(base_url: str) -> list[dict]:
	"""Fetch and chunk every SITS function reference page.

	Args:
		base_url (str): The base URL of the SITS function reference.

	Returns:
		list[dict]: The list of function chunk dictionaries.

	Raises:
		httpx.HTTPError: If the sitemap cannot be fetched.
	"""
	records = []

	# define client
	with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
		# fetch the sitemap URLs
		urls = py_.filter_(
			_fetch_sitemap_urls(client, base_url),
			lambda u: "/reference/" in u and not u.endswith("/reference/index.html"),
		)

		# iterate over the URLs
		for url in tqdm(urls, desc="Parsing SITS function reference", unit="page"):
			# fetch the page
			try:
				response = client.get(url)
				response.raise_for_status()

			except httpx.HTTPError as error:
				tqdm.write(f"Skipping {url}: {error}")

				continue

			# parse the page
			record = _parse_reference_page(
				url=url,
				html=response.text,
			)

			if record:
				records.append(record)

	return records


def parse_references(references_path: Path) -> list[dict]:
	"""Build reference article records.

	Args:
		references_path (Path): Path to the ``references.json`` file.

	Returns:
		list[dict]: One chunk dictionary per reference article.

	Raises:
		FileNotFoundError: If ``references_path`` does not exist.
	"""
	chunks = []
	entries = orjson.loads(references_path.read_bytes())

	# iterate over the entries
	for entry in entries:
		url = py_.get(entry, "url")

		# skip entries we could not cite
		if not url:
			continue

		# build the record
		chunks.append(
			{
				"id": _chunk_id("ref", py_.get(entry, "id", url)),
				"text": _render_reference(entry),
				"source": "reference_article",
				"title": py_.get(entry, "title", "Untitled"),
				"url": url,
				"function_name": None,
			}
		)

	# return!
	return chunks
