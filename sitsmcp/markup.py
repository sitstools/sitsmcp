#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Quarto HTML to Markdown module."""

import re

import pydash as py_
from bs4 import NavigableString, PageElement, Tag

#
# Constants
#
MAIN_LANGUAGE = "R"
"""Tab label the main text keeps from an R/Python tabset."""

EXTRA_LANGUAGES = ("Python",)
"""Tab labels indexed as chunks of their own."""

CODE_LANGUAGES = {MAIN_LANGUAGE, *EXTRA_LANGUAGES}
"""Tab labels that make a tabset a language tabset."""

SKIPPED_TAGS = {"section", "h1", "h2", "h3", "script", "style", "button", "nav", "img"}
"""Tags that never reach the Markdown."""

BLOCK_TAGS = {"p", "div", "pre", "figure", "table", "ul", "ol", "blockquote"}
"""Tags that end a run of inline text."""


#
# Internal functions - inline text
#
def _inline(node: PageElement) -> str:
	"""Convert an inline node to Markdown.

	Args:
		node (PageElement): A text node or an inline element.

	Returns:
		str: The Markdown.
	"""
	# text
	if isinstance(node, NavigableString):
		return str(node)

	if not isinstance(node, Tag) or node.name in SKIPPED_TAGS:
		return ""

	# inline code
	if node.name == "code":
		return f"`{collapse(node.get_text())}`"

	# children
	text = "".join(_inline(child) for child in node.children)

	# a link with no text, such as a badge image, keeps only its spaces
	if node.name == "a" and not collapse(text):
		return text

	# absolute links only
	href = str(node.get("href", ""))
	if node.name == "a" and href.startswith(("http://", "https://")):
		return f"[{collapse(text)}]({href})"

	return text


#
# Internal functions - blocks
#
def _is_list(node: PageElement) -> bool:
	"""Tell whether a node is a `ul` or `ol` element."""
	return isinstance(node, Tag) and node.name in ("ul", "ol")


def _is_tabset(node: Tag) -> bool:
	"""Tell whether an element is a Quarto tabset (`div.panel-tabset`)."""
	return node.name == "div" and "panel-tabset" in (node.get("class") or [])


def _code_language(pre: Tag) -> str:
	"""Return the language of a code block.

	Args:
		pre (Tag): The `pre` element of a Quarto code block.

	Returns:
		str: The language, such as `"r"`, or an empty string if it has none.
	"""
	# get the classes
	classes = py_.map_(pre.get("class") or [], str)

	# get language
	if "sourceCode" not in classes:
		return ""

	# get language classes
	after = classes[classes.index("sourceCode") + 1 :]

	# assume the first non-code class is the language
	return py_.find(after, lambda name: not name.startswith("code-")) or ""


def _block_code(pre: Tag) -> str:
	"""Convert a `pre` element to a fenced block.

	Args:
		pre (Tag): The `pre` element of a Quarto code block.

	Returns:
		str: The fenced block, in the language of the code.
	"""
	# get the code, without the copy button
	code = "".join(
		string for string in pre.strings if string.find_parent("button") is None
	)

	# return the fenced block
	return f"```{_code_language(pre)}\n{code.rstrip()}\n```"


def _table_cells(row: Tag) -> list[str]:
	"""Return the Markdown of the cells of a table row, with pipes escaped."""
	return [collapse(_inline(cell)).replace("|", "\\|") for cell in row(["th", "td"])]


def _pipe_row(cells: list[str]) -> str:
	"""Write the cells of a row as one Markdown table row."""
	return f"| {' | '.join(cells)} |"


def _block_table(table: Tag) -> str:
	"""Convert a `table` element to pipe rows.

	Args:
		table (Tag): The `table` element.

	Returns:
		str: The Markdown table, or an empty string if it has no rows.
	"""
	rows = py_.map_(table("tr"), _table_cells)
	if not rows:
		return ""

	# the first row is the header, followed by the separator row
	header, *body = rows
	separator = ["---"] * len(header)

	# join the rows
	return "\n".join(py_.map_([header, separator, *body], _pipe_row))


def _item_text(item: Tag) -> str:
	"""Return the Markdown of a list item, without the lists nested in it."""
	return collapse(
		"".join(_inline(child) for child in item.children if not _is_list(child))
	)


def _block_list(items: Tag, indent: str = "") -> str:
	"""Convert a `ul` or `ol` element to Markdown items.

	Args:
		items (Tag): The list element.

		indent (str): Prefix of the lines, for nested lists.

	Returns:
		str: The Markdown items, one per line.
	"""
	lines = []

	for number, item in enumerate(items("li", recursive=False), start=1):
		# get the bullet
		bullet = f"{number}." if items.name == "ol" else "-"

		# add the item to the list
		lines.append(f"{indent}{bullet} {_item_text(item)}")
		lines.extend(
			_block_list(nested, indent + "  ")
			for nested in item(["ul", "ol"], recursive=False)
		)

	# join the lines
	return "\n".join(lines)


def _block_figure(figure: Tag) -> str:
	"""Convert a `figure` element to its caption (empty if it has none)."""
	caption = figure.find("figcaption")

	return collapse(_inline(caption)) if isinstance(caption, Tag) else ""


def _block_paragraph(paragraph: Tag) -> str:
	"""Convert a `p` element to one line of text."""
	return collapse(_inline(paragraph))


BLOCK_CONVERTERS = {
	"pre": _block_code,
	"table": _block_table,
	"ul": _block_list,
	"ol": _block_list,
	"figure": _block_figure,
	"p": _block_paragraph,
}
"""Block tags that become one Markdown block; the others are walked."""


#
# Internal functions - tabsets
#
def _tab_labels(tabset: Tag) -> list[str]:
	"""Return the labels of a tabset, not those of nested tabsets."""
	nav = tabset.find("ul", recursive=False)
	links = nav("a", class_="nav-link") if nav else []

	# get the labels
	return [collapse(link.get_text()) for link in links]


def _tab_panes(tabset: Tag) -> list[Tag]:
	"""Return the panes of a tabset, not those of nested tabsets."""
	content = tabset.find("div", class_="tab-content", recursive=False)

	# get the panes
	return content("div", class_="tab-pane", recursive=False) if content else []


def _tabset(tabset: Tag) -> tuple[list[tuple[str, Tag]], bool]:
	"""Pair the labels of a tabset with their own panes.

	Args:
		tabset (Tag): A `div.panel-tabset` element.

	Returns:
		tuple[list[tuple[str, Tag]], bool]: The `label`, `pane` pairs, in order,
			and whether it is a language tabset.
	"""
	# get the labels and panes
	labels = _tab_labels(tabset)
	panes = _tab_panes(tabset)

	# tell whether it is a language tabset
	is_code = (
		bool(labels) and len(labels) == len(panes) and set(labels) <= CODE_LANGUAGES
	)

	# a pane with no label of its own gets an empty label
	pane_labels = labels + [""] * (len(panes) - len(labels))

	return list(zip(pane_labels, panes)), is_code


def _block_tabset(
	tabset: Tag, languages: tuple[str, ...], only: str | None
) -> list[str]:
	"""Convert a language tabset to Markdown blocks.

	Args:
		tabset (Tag): A `div.panel-tabset` element.

		languages (tuple[str, ...]): Tab labels kept from the language tabsets.

		only (str | None): When set, only the panes with this label are kept.

	Returns:
		list[str]: The Markdown blocks, one per pane.
	"""
	panes, is_code = _tabset(tabset)

	# keep only the panes of the languages asked for
	if is_code:
		# get the languages to keep
		keep = (only,) if only else languages
		kept = [pane for label, pane in panes if label in keep]

		# get children blocks of the given panes
		return py_.flat_map(kept, lambda pane: _walk(pane, languages))

	# other language tabsets keep every pane, under its label
	blocks = []

	for label, pane in panes:
		# add the label to the blocks
		if only is None and label:
			blocks.append(f"**{label}**")

		# add the children blocks of the given pane
		blocks.extend(_walk(pane, languages, only))

	return blocks


#
# Internal functions - walking
#
def _finished_run(run: str) -> list[str]:
	"""Return a run of inline text as a Markdown block, or nothing if it is blank."""
	text = collapse(run)

	# return the run as a block, or nothing if it is blank
	return [text] if text else []


def _walk(
	parent: Tag, languages: tuple[str, ...], only: str | None = None
) -> list[str]:
	"""Walk the children of an element, collecting Markdown blocks from its content.

	Args:
		parent (Tag): The element to walk.

		languages (tuple[str, ...]): Tab labels kept from the language tabsets.

		only (str | None): When set, only the panes with this label are kept.

	Returns:
		list[str]: The Markdown blocks, one per line.
	"""
	blocks = []
	run = ""

	# iterate over the children
	for child in parent.children:
		# inline text builds a run, ended by the next block tag
		if not isinstance(child, Tag) or child.name not in BLOCK_TAGS:
			run += _inline(child) if only is None else ""

			continue

		# add the run as a block
		blocks.extend(_finished_run(run))
		run = ""

		# add the block to the blocks
		blocks.extend(_block(child, languages, only))

	# add the run as a block
	blocks.extend(_finished_run(run))

	# return the blocks
	return blocks


def _block(node: Tag, languages: tuple[str, ...], only: str | None) -> list[str]:
	"""Convert a block element to Markdown blocks from its content.

	Args:
		node (Tag): The block element.

		languages (tuple[str, ...]): Tab labels kept from the language tabsets.

		only (str | None): When set, only the panes with this label are kept.

	Returns:
		list[str]: The Markdown blocks, one per line.
	"""
	# convert a language tabset to Markdown blocks
	if _is_tabset(node):
		return _block_tabset(node, languages, only)

	# only language tabsets are kept when a label is asked for
	if only is not None:
		return (
			_walk(node, languages, only) if node.name in ("div", "blockquote") else []
		)

	# a block that holds other blocks, such as a `div`, is walked
	convert = BLOCK_CONVERTERS.get(node.name)
	if convert is None:
		return _walk(node, languages)

	# convert the block to Markdown blocks
	text = convert(node)

	# return the blocks
	return [text] if text else []


#
# Public functions
#
def collapse(text: str) -> str:
	"""Collapse every run of whitespace into a single space.

	Args:
		text (str): Text of a prose element or a heading.

	Returns:
		str: The collapsed text, stripped.
	"""
	return re.sub(r"\s+", " ", text).strip()


def to_markdown(element: Tag, languages: tuple[str, ...] = (MAIN_LANGUAGE,)) -> str:
	"""Convert the content of a Quarto element to Markdown blocks.

	Nested sections and `h1`-`h3` headings are left out.

	Args:
		element (Tag): The element, such as a `section`.

		languages (tuple[str, ...]): Tab labels kept from the language tabsets.

	Returns:
		str: The Markdown blocks, one per line, joined by blank lines.
	"""
	return "\n\n".join(_walk(element, languages))


def tab_markdown(element: Tag, label: str) -> str:
	"""Convert only the panes of one label of the language tabsets.

	Args:
		element (Tag): The element, such as a `section`.

		label (str): The tab label, such as `"Python"`.

	Returns:
		str: The Markdown blocks, one per line, joined by blank lines (empty if none).
	"""
	return "\n\n".join(_walk(element, (label,), only=label))
