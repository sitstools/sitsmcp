#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Data preparation module."""

import copy
import hashlib
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import httpx
import orjson
import pydash as py_
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from sitsmcp.markup import (
	EXTRA_LANGUAGES,
	MAIN_LANGUAGE,
	collapse,
	tab_markdown,
	to_markdown,
)

#
# Constants
#
HTTP_TIMEOUT = 30.0
"""Per-request timeout, in seconds."""

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
"""XML namespace used by ``sitemap.xml`` documents."""

CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]
"""Ordered separators used by the recursive chunker."""

VERSION_RE = re.compile(r"^\d+(\.\d+)+$")
"""Package version format (e.g., ``1.5.4``)."""

LAST_BUILT_RE = re.compile(r"^last_built:\s*(\S+)", re.MULTILINE)
"""The ``last_built`` line of a Quarto site's ``quarto.yml``."""

FUNCTION_SKIPPED_SECTIONS = ("author", "references")
"""Sections of a reference page that the search does not index."""

INTERNAL_FUNCTION_PREFIX = "dot-"
"""Quarto file-name prefix for internal functions (``.name`` becomes ``dot-name``)."""

SectionUnit = tuple[str | None, str | None, str | None, Tag]
"""A book unit: section ID, number, title and its ``<section>`` tag."""


#
# Internal functions - chunking
#
def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
	"""Prepend the tail of each chunk to the next one.

	The tail starts at a word boundary, so it is shorter than the chunk overlap.

	Args:
		chunks (list[str]): The ordered list of chunks.

		chunk_overlap (int): Maximum number of trailing characters to
			carry over between chunks.

	Returns:
		list[str]: The list of chunks with overlap applied.
	"""
	# prepend the tail of each chunk to the next one
	result = [chunks[0]]

	for index in range(1, len(chunks)):
		# get the previous chunk "tail"
		prev_tail = chunks[index - 1][-chunk_overlap:]

		# start the tail at a word boundary
		space = prev_tail.find(" ")
		prev_tail = prev_tail[space + 1 :] if space >= 0 else ""

		# prepend the previous chunk "tail" to the current
		# chunk and strip whitespace
		result.append((prev_tail + " " + chunks[index]).strip())

	# return!
	return result


def _hard_split(text: str, chunk_size: int) -> list[str]:
	"""Split text at fixed character boundaries when no separator applies.

	Args:
		text (str): The text to split.

		chunk_size (int): The maximum size of each chunk.

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
		start = end

	# return
	return chunks


def _split_with_separator(
	text: str, sep: str, chunk_size: int, separators: list[str]
) -> list[str]:
	"""Split on a separator and merge pieces into chunks of the target size.

	Args:
		text (str): The text to split.

		sep (str): The separator to split on.

		chunk_size (int): The maximum size of each chunk.

		separators (list[str]): The remaining separator ladder for recursion.

	Returns:
		list[str]: The list of chunks.
	"""
	pieces = text.split(sep)

	# re-attach the period on sentence splits, and join them with a space
	if sep == ". ":
		# add the period to the last piece
		pieces = [
			piece + "." if index < len(pieces) - 1 else piece
			for index, piece in enumerate(pieces)
		]

		# join the pieces with a space
		join = " "

	else:
		join = sep

	# initialize the current chunk
	current = ""
	chunks = []

	# iterate over the pieces
	for piece in pieces:
		# get the candidate chunk
		candidate = (current + join + piece).strip() if current else piece.strip()

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
			chunks.extend(_recursive_split(piece.strip(), remaining, chunk_size))

			# reset the current chunk
			current = ""

		else:
			# reset the current chunk
			current = piece.strip()

	# add the current chunk to the list
	if current:
		chunks.append(current)

	# return chunks
	return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
	"""Split text using the first separator that applies.

	The split carries no overlap. The chunker applies it once at the top.

	Args:
		text (str): The text to split.

		separators (list[str]): The ordered separator ladder.

		chunk_size (int): The maximum size of each chunk.

	Returns:
		list[str]: The list of chunks.
	"""
	if len(text) <= chunk_size:
		return [text]

	for sep in separators:
		if sep in text:
			return _split_with_separator(text, sep, chunk_size, separators)

	# if no separator applies, split at fixed character boundaries
	return _hard_split(text, chunk_size)


def _markdown_blocks(markdown: str) -> list[str]:
	"""Split Markdown on blank lines.

	Args:
		markdown (str): The Markdown to split.

	Returns:
		list[str]: The blocks, without empty ones.
	"""
	blocks = []
	lines = []
	is_fenced = False

	for line in markdown.strip().split("\n"):
		# check if the line is a fenced code block
		if line.startswith("```"):
			is_fenced = not is_fenced

		# check if the line is a blank line
		if not is_fenced and not line.strip():
			# add the lines to the blocks
			if lines:
				blocks.append("\n".join(lines))
				lines = []

			# continue to the next line
			continue

		# add the line to the lines
		lines.append(line)

	# add the last lines to the blocks
	if lines:
		blocks.append("\n".join(lines))

	# return
	return blocks


def _fit_block(block: str, chunk_size: int) -> list[str]:
	"""Split a block that is larger than the chunk size.

	Args:
		block (str): A prose block or a whole fenced code block.

		chunk_size (int): The maximum size of each piece.

	Returns:
		list[str]: The pieces, each of at most ``chunk_size`` characters.
	"""
	if len(block) <= chunk_size:
		return [block]

	# split the block into lines
	lines = block.split("\n")

	# prose is split as text
	if not (lines[0].startswith("```") and lines[-1] == "```"):
		return chunk_text(block, chunk_size, 0)

	# code is split on lines, fenced again with the same language
	opening, closing = lines[0], "```"
	room = max(chunk_size - len(opening) - len(closing) - 2, 1)

	size = 0
	pieces = []
	current = []

	# iterate over the lines
	for line in lines[1:-1]:
		# a line larger than the room is cut into parts
		parts = [line[start : start + room] for start in range(0, len(line), room)]
		parts = parts or [""]

		# process parts
		for part in parts:
			# if the current chunk is larger than the room, add the current
			# chunk to the pieces
			if current and size + len(part) + 1 > room:
				# add the current chunk to the pieces
				pieces.append("\n".join([opening, *current, closing]))

				# reset the current chunk
				current = []
				size = 0

			# add the part to the current chunk
			current.append(part)
			size += len(part) + 1

	# add the last chunk to the pieces
	if current:
		pieces.append("\n".join([opening, *current, closing]))

	# return
	return pieces


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
def _extract_chapter(soup: BeautifulSoup) -> tuple[str | None, str | None]:
	"""Extract the chapter number and title.

	Extract the page/chapter title from an ``<h1>`` or ``<title>`` tag.

	Args:
		soup (BeautifulSoup): The parsed page.

	Returns:
		tuple[str | None, str | None]: The chapter number and the chapter title,
		either of which is `None` if absent.
	"""
	heading = soup.select_one("h1.title") or soup.find("h1")

	# split the heading, if Quarto numbered it
	if isinstance(heading, Tag):
		number = heading.select_one(".chapter-number")
		title = heading.select_one(".chapter-title")

		if number and title:
			return collapse(number.get_text()), collapse(title.get_text())

	# otherwise, use the text of the heading or of <title>
	element = heading or soup.find("title")

	# return
	return None, (collapse(element.get_text()) if element else None) or None


def _section_title(section: Tag) -> str | None:
	"""Extract the title of a section, without its number.

	Args:
		section (Tag): A ``<section>`` of the page.

	Returns:
		str | None: The heading text, or `None` if the section has no heading.
	"""
	heading = section.find(["h2", "h3", "h4"], recursive=False)

	if not isinstance(heading, Tag):
		return None

	# drop the number written inside the heading
	heading = copy.copy(heading)

	for number in heading.select(".header-section-number"):
		number.decompose()

	# return
	return collapse(heading.get_text()) or None


def _section_units(main: Tag) -> list[SectionUnit]:
	"""List the units of a page.

	The bibliography and sections without an ID are left out.

	Args:
		main (Tag): The page's main content.

	Returns:
		list[SectionUnit]: The ID, number, title and tag of each unit; the
		introduction has no ID, number or title.
	"""
	sections = []

	# iterate over the sections in document order
	for section in main.select("section.level2, section.level3, section.level4"):
		section_id = section.get("id")

		# skip the bibliography and sections without an ID
		if not section_id or section_id == "references":
			continue

		# get the section number
		number = section.get("data-number")

		# add the section to the list
		sections.append(
			(
				str(section_id),
				str(number) if number else None,
				_section_title(section),
				section,
			)
		)

	# return the introduction first
	return [(None, None, None, main), *sections]


def _book_sections(main: Tag) -> list[dict]:
	"""Split the main content of a page into its sections.

	Args:
		main (Tag): The page's main content.

	Returns:
		list[dict]: Dictionaries with the section ID, number, title,
		its main Markdown and extra Markdown of each extra language.
	"""
	sections = []

	# iterate over the sections
	for section_id, number, title, element in _section_units(main):
		# convert
		markdown = to_markdown(element=element, languages=(MAIN_LANGUAGE,))

		# drop Markdown with no word characters
		if not re.search(r"\w", markdown):
			markdown = ""

		# get the extra Markdown
		extra = {
			label: text
			for label in EXTRA_LANGUAGES
			if (text := tab_markdown(element=element, label=label))
		}

		# skip if there is no Markdown
		if not markdown and not extra:
			continue

		# add the section to the list
		sections.append(
			{
				"id": section_id,
				"number": number,
				"title": title,
				"markdown": markdown,
				"extra": extra,
			}
		)

	# return!
	return sections


def _parse_book_page(
	url: str, html: str, chunk_size: int, chunk_overlap: int
) -> list[dict]:
	"""Extract and chunk the main content of a SITS book page.

	Args:
		url (str): The page URL.

		html (str): The raw HTML.

		chunk_size (int): The maximum size of each chunk.

		chunk_overlap (int): Ignored, kept for the signature.

	Returns:
		list[dict]: The list of chunk dictionaries for the page.
	"""
	# parse
	soup = BeautifulSoup(html, "lxml")
	main = soup.find("main") or soup.find("div", {"class": "content"})

	# skip if not a tag
	if not isinstance(main, Tag):
		return []

	# extract chapter
	chapter_number, chapter_title = _extract_chapter(soup)

	chapter_label = (
		f"Chapter {chapter_number} {chapter_title}"
		if chapter_number and chapter_title
		else chapter_title
	)

	# iterate over the sections
	chunks = []

	for section in _book_sections(main):
		# get the section ID
		section_id = section["id"]

		# get the section title
		title = (
			" ".join(py_.compact([section["number"], section["title"]]))
			or chapter_title
		)

		# get the main chunks and the chunks of each extra language
		texts = [(None, section["markdown"])] + [
			(extra_label, text) for extra_label, text in section["extra"].items()
		]

		# iterate over the chunks
		for language_label, markdown in texts:
			suffix = f" ({language_label})" if language_label else ""

			# get the context
			context = " › ".join(
				py_.compact([chapter_label, title if section_id else None])
			)
			context = context or title or ""

			# iterate over the chunks
			for index, text in enumerate(chunk_markdown(markdown, chunk_size)):
				language = language_label.lower() if language_label else None

				# set the language to R if the chunk starts with ```r\n
				if language is None and "```r\n" in text:
					language = MAIN_LANGUAGE.lower()

				# create the chunk dictionary
				chunks.append(
					{
						"id": _chunk_id(
							"book", url, section_id or "", language or "", str(index)
						),
						"text": f"{context}{suffix}\n\n{text}",
						"source": "book",
						"title": title,
						"url": f"{url}#{section_id}" if section_id else url,
						"function_name": None,
						"page_url": url,
						"chapter_number": chapter_number,
						"chapter_title": chapter_title,
						"section_id": section_id,
						"section_number": section["number"],
						"section_title": section["title"],
						"language": language,
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


def _is_public_reference(url: str) -> bool:
	"""Check whether a sitemap URL is the reference page of a public function.

	Args:
		url (str): A URL from the sitemap.

	Returns:
		bool: ``True`` for a reference page other than the index and the
			dot-prefixed internal functions.
	"""
	if "/reference/" not in url or url.endswith("/reference/index.html"):
		return False

	# return!
	return not _function_name_from_url(url).startswith(INTERNAL_FUNCTION_PREFIX)


def _reference_version(soup: BeautifulSoup) -> str | None:
	"""Read the package version a reference page shows in its navigation bar.

	Args:
		soup (BeautifulSoup): The parsed reference page.

	Returns:
		str | None: The version, or ``None`` if the page shows none.
	"""
	# get the version element
	small = soup.select_one("small.nav-text")
	if small is None:
		return None

	# keep only something that looks like a version number
	version = small.get_text(strip=True)

	# return!
	return version if VERSION_RE.match(version) else None


def _reference_arguments(section: Tag, url: str) -> list[dict]:
	"""Read the arguments of a reference page's "Arguments" section.

	Args:
		section (Tag): The section holding the argument list.

		url (str): The reference page URL.

	Returns:
		list[dict]: One {"name", "url", "text"} dict per argument.
	"""
	arguments = []
	# iterate over the argument names
	for dt in section.find_all("dt"):
		# get the description that follows the name
		dd = dt.find_next_sibling("dd")

		# skip if there is no description
		if dd is None:
			continue

		# link to the argument when the page gives it an anchor
		anchor = dt.get("id")

		# add the argument
		arguments.append(
			{
				"name": dt.get_text(strip=True),
				"url": f"{url}#{anchor}" if anchor else url,
				"text": to_markdown(dd),
			}
		)

	# return!
	return arguments


def _reference_sections(main: Tag, url: str) -> tuple[list[dict], list[dict]]:
	"""Split a reference page into sections and arguments.

	Args:
		main (Tag): The reference page's main content.

		url (str): The reference page URL.

	Returns:
		tuple[list[dict], list[dict]]: The sections, in page order and starting
										with the description, and the arguments.
	"""
	sections = []
	arguments = []

	# read the description, which has no heading
	description = main.find(class_="ref-description")
	if isinstance(description, Tag):
		# get the description Markdown
		text = to_markdown(description)

		# skip if there is no Markdown
		if text:
			# add the description to the sections
			sections.append(
				{"id": None, "name": "Description", "url": url, "text": text}
			)

	# iterate over the headings
	for heading in main.find_all("h2"):
		# get the section
		section = heading.parent
		section_id = heading.get("id")

		# skip a heading without a section around it
		if section is None or not isinstance(section_id, str | None):
			continue

		# get the arguments as a list, the other sections as they are
		if section_id == "arguments":
			# get the arguments
			arguments = _reference_arguments(section, url)

			# indent the later lines of an argument, not its blank lines
			# (functional way is the right way)
			text = "\n".join(
				f"- `{argument['name']}`: "
				+ "\n".join(
					f"  {line}" if line and index else line
					for index, line in enumerate(argument["text"].split("\n"))
				)
				for argument in arguments
			)

		else:
			text = to_markdown(section)

		# drop the empty sections
		if not text:
			continue

		# add the section
		sections.append(
			{
				"id": section_id,
				"name": heading.get_text(strip=True),
				"url": f"{url}#{section_id}" if section_id else url,
				"text": text,
			}
		)

	# return!
	return sections, arguments


def _parse_reference_page(url: str, html: str) -> dict | None:
	"""Build the record of a function reference page.

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

	# extract the version and the sections
	version = _reference_version(soup)
	sections, arguments = _reference_sections(main, url)

	# join the sections into the body
	body = "\n\n".join(f"{section['name']}:\n{section['text']}" for section in sections)

	# skip if the body is empty
	if not body:
		return None

	# create the text
	header = f"Function: {function_name}\nTitle: {title}\nDocs: {url}"
	if version:
		header += f"\nVersion: {version}"

	# create the text
	text = f"{header}\n\n{body}"

	# create the record
	return {
		"id": _chunk_id("function", function_name),
		"text": text,
		"source": "function",
		"title": title,
		"url": url,
		"function_name": function_name,
		"version": version,
		"sections": sections,
		"arguments": arguments,
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
	Each chunk is at most ``chunk_size`` characters, overlap included, with up
	to ``chunk_overlap`` characters carried over from the previous chunk.

	Args:
		text (str): The text to split.

		chunk_size (int): The maximum size of each chunk.

		chunk_overlap (int): The overlap between consecutive chunks.

	Returns:
		list[str]: The list of chunks.
	"""
	if not text or not text.strip():
		return []

	# split with no overlap, leaving room for the overlap
	chunks = _recursive_split(
		text=text.strip(),
		separators=CHUNK_SEPARATORS,
		chunk_size=max(chunk_size - chunk_overlap, 1),
	)

	# apply the overlap once, if required
	if chunk_overlap > 0 and len(chunks) > 1:
		chunks = _apply_overlap(chunks=chunks, chunk_overlap=chunk_overlap)

	# return!
	return chunks


def chunk_markdown(markdown: str, chunk_size: int) -> list[str]:
	"""Split Markdown into chunks without breaking fenced code blocks.

	Splits on blank lines outside fences and packs the blocks into chunks of at
	most ``chunk_size`` characters. A fenced block only larger than
	``chunk_size`` is split on lines, each piece fenced again with the same
	language; a prose block larger than it is split as text, without overlap.

	Args:
		markdown (str): The Markdown to split.

		chunk_size (int): The maximum size of each chunk.

	Returns:
		list[str]: The list of chunks.
	"""
	chunks = []
	current = ""

	# pack the blocks
	for block in _markdown_blocks(markdown):
		# iterate over the pieces
		for piece in _fit_block(block, chunk_size):
			# join the pieces
			joined = f"{current}\n\n{piece}" if current else piece

			# add the joined piece to the chunks
			if len(joined) <= chunk_size:
				current = joined
				continue

			if current:
				chunks.append(current)

			current = piece

	if current:
		chunks.append(current)

	# return!
	return chunks


def function_chunks(records: list[dict], chunk_size: int) -> list[dict]:
	"""Turn function reference records into search chunks.

	Args:
		records (list[dict]): The records.

		chunk_size (int): The maximum size of each chunk.

	Returns:
		list[dict]: The list of chunk dictionaries.
	"""
	chunks = []

	# define the skipped sections
	skipped = (*FUNCTION_SKIPPED_SECTIONS, "arguments")

	# iterate over the records
	for record in records:
		# get the function name and page URL
		name = record["function_name"]
		page_url = record["url"]
		title = f"{name}: {record['title']}"

		# create units using record sections
		units = [
			(
				section["id"],
				section["id"]
				or ("" if section["name"] == "Description" else f"section-{index}"),
				section["name"],
				section["url"],
				section["text"],
			)
			for index, section in enumerate(record["sections"])
			if section["id"] not in skipped
		]

		# create units using record arguments
		for index, argument in enumerate(record["arguments"]):
			# get the anchor
			anchor = (
				argument["url"].split("#", 1)[1] if "#" in argument["url"] else None
			)

			# add the unit to the list
			units.append(
				(
					anchor,
					anchor or f"argument-{index}",
					f"Argument {argument['name']}",
					argument["url"],
					argument["text"],
				)
			)

		# iterate over the units
		for section_id, id_part, section_title, url, markdown in units:
			# iterate over the pieces
			for index, piece in enumerate(chunk_markdown(markdown, chunk_size)):
				# create the chunk dictionary
				chunks.append(
					{
						"id": _chunk_id("function", name, id_part, str(index)),
						"text": f"{title} › {section_title}\n\n{piece}",
						"source": "function",
						"title": title,
						"url": url,
						"function_name": name,
						"page_url": page_url,
						"chapter_number": None,
						"chapter_title": None,
						"section_id": section_id,
						"section_number": None,
						"section_title": section_title,
						"language": "r" if "```r\n" in piece else None,
					}
				)

	# return!
	return chunks


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
		urls = py_.filter_(_fetch_sitemap_urls(client, base_url), _is_public_reference)

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


def normalize_timestamp(value: str) -> str | None:
	"""Write a timestamp as "YYYY-MM-DDTHH:MM:SSZ", in UTC.

	A time with no zone is taken as UTC.

	Args:
		value (str): An ISO 8601 timestamp, such as "2026-10-03T14:48:01.133Z".

	Returns:
		str | None: The normalised timestamp, or `None` if it cannot be parsed.
	"""
	# parse the timestamp
	try:
		parsed = datetime.fromisoformat(value)

	except ValueError:
		return None

	# set the timezone to UTC
	parsed = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed

	# return!
	return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_book_build(client: httpx.Client, base_url: str) -> str | None:
	"""Read when the SITS book was last built, from its sitemap.

	Args:
		client (httpx.Client): An open HTTP client.

		base_url (str): The base URL of the SITS book.

	Returns:
		str | None: The newest `lastmod` of the sitemap, normalised, or
			`None` if no page has a parseable one, or if the answer is not XML.

	Raises:
		httpx.HTTPError: If the sitemap cannot be fetched.
	"""
	# request
	response = client.get(f"{base_url}/sitemap.xml")
	response.raise_for_status()

	# read the lastmod of every page
	try:
		root = ElementTree.fromstring(response.text)

	except ElementTree.ParseError:
		return None

	mods = [
		normalize_timestamp((lastmod.text or "").strip())
		for lastmod in root.findall(".//sm:lastmod", SITEMAP_NS)
	]

	# return! (the normal form sorts as time)
	return max(py_.compact(mods), default=None)


def fetch_reference_build(client: httpx.Client, base_url: str) -> str | None:
	"""Read when the SITS function reference was last built.

	Args:
		client (httpx.Client): An open HTTP client.

		base_url (str): The base URL of the SITS function reference.

	Returns:
		str | None: The `last_built` of the site, normalised, or `None` if
			the file has no parseable one.

	Raises:
		httpx.HTTPError: If the file cannot be fetched.
	"""
	# request
	response = client.get(f"{base_url}/pkgdown.yml")
	response.raise_for_status()

	# get the line
	found = LAST_BUILT_RE.search(response.text)

	# return!
	return normalize_timestamp(found.group(1)) if found else None


def reference_version(records: list[dict]) -> str | None:
	"""Pick the package version of a reference.

	Args:
		records (list[dict]): The reference records, each with a `version`.

	Returns:
		str | None: The version, or `None` if no record has one.
	"""
	counts = Counter(py_.compact([record.get("version") for record in records]))

	# return!
	return max(
		counts,
		key=lambda version: (
			counts[version],
			[int(part) for part in version.split(".")],
		),
		default=None,
	)


def parse_references(references_path: Path) -> list[dict]:
	"""Build reference chunk records.

	Args:
		references_path (Path): Path to the `references.json` file.

	Returns:
		list[dict]: One chunk dictionary per reference.

	Raises:
		FileNotFoundError: If `references_path` does not exist.
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
