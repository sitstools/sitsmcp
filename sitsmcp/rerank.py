#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsmcp is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sitsmcp - Rerank module."""

import logging
import threading
from typing import Any, Protocol

import httpx
import pydash as py_
from pydantic import SecretStr
from sqlalchemy import Engine

from sitsmcp import store
from sitsmcp.config import Settings

#
# Constants
#
logger = logging.getLogger(__name__)
"""Logger of the module, ``sitsmcp.rerank``."""

RERANKER_PROVIDERS = ("cohere", "fastembed")
"""Providers a reranker setting may name."""

LOCAL_PROVIDERS = ("fastembed",)
"""Providers that run in the process and cost nothing per call."""

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
"""Endpoint of the Cohere v2 rerank API."""

_shared_rerankers = {}
"""The rerankers built so far in this process."""

_shared_lock = threading.Lock()
"""Guards `_shared_rerankers`, so two threads build one reranker."""


#
# Exceptions
#
class RerankError(Exception):
	"""A reranker could not order the documents."""


#
# Internal functions
#
def _load_cross_encoder(model: str) -> Any:
	"""Load a fastembed cross-encoder.

	Args:
		model (str): The fastembed model name.

	Returns:
		Any: The ``TextCrossEncoder``.
	"""
	from fastembed.rerank.cross_encoder import TextCrossEncoder

	# return encoder
	return TextCrossEncoder(model_name=model)


def _cohere_key(settings: Settings) -> SecretStr | None:
	"""Return the Cohere key, or ``None`` when it is unset or empty."""
	key = settings.reranker_api_key

	# check if the key is None or empty
	if key is None or not key.get_secret_value():
		return None

	# return!
	return key


def _is_active(spec: str, settings: Settings) -> bool:
	"""Say whether a reranker setting turns a reranker on."""
	if not spec:
		return False

	# parse the reranker model
	provider, _ = parse_reranker_model(spec)

	# check if the provider is not cohere or the key is not None
	return provider != "cohere" or _cohere_key(settings) is not None


def _is_key_client(client: str | None) -> bool:
	"""Say whether a client gets the key reranker."""
	return client is None or client.startswith("key:")


def _is_local(spec: str) -> bool:
	"""Say whether a reranker setting names a local model, which costs nothing."""
	if not spec:
		return False

	# parse the reranker model
	provider, _ = parse_reranker_model(spec)

	# return!
	return provider in LOCAL_PROVIDERS


def _cache_key_of(spec: str, settings: Settings) -> tuple:
	"""Key a local reranker by its spec alone, and Cohere by its key and timeout."""
	if _is_local(spec):
		return (spec,)

	# return!
	return (spec, settings.reranker_api_key, settings.reranker_timeout)


def _shared_reranker(spec: str, settings: Settings) -> "Reranker | None":
	"""Build the reranker of a spec once per process, and return it."""
	cache_key = _cache_key_of(spec, settings)

	with _shared_lock:
		if cache_key not in _shared_rerankers:
			_shared_rerankers[cache_key] = create_reranker(spec, settings)

		# return!
		return _shared_rerankers[cache_key]


def _ranking_of(
	reranker: "Reranker", query: str, hits: list[dict]
) -> list[tuple[int, float]] | None:
	"""Order the hits with one reranker.

	Args:
		reranker (Reranker): The reranker to try.

		query (str): The search query.

		hits (list[dict]): The fused candidates.

	Returns:
		list[tuple[int, float]] | None: The ranking, or ``None`` on a failure.
	"""
	try:
		return reranker.rerank(query, [hit["text"] for hit in hits])

	except RerankError as error:
		logger.warning("Reranking with %s failed: %s", reranker.name, error)
		return None


def _status_of_key_reranker(settings: Settings) -> str:
	"""Describe the key reranker."""
	spec = settings.key_reranker_model

	# check if the spec is None or empty
	if not spec:
		return "none"

	# return the name, or the reason
	if _is_active(spec, settings):
		return spec

	# return the reason
	return f"off ({spec} needs SITSMCP_RERANKER_API_KEY)"


#
# Public functions
#
def parse_reranker_model(spec: str) -> tuple[str, str]:
	"""Split a reranker spec into provider and model.

	Args:
		spec (str): The reranker spec, in format ``<provider>/<model>``.

	Returns:
		tuple[str, str]: The provider and the model.

	Raises:
		ValueError: The provider is unknown or the model is empty.
	"""
	provider, _, model = spec.partition("/")

	# check provider
	if provider not in RERANKER_PROVIDERS:
		raise ValueError(
			f"Unknown reranker provider {provider!r} in {spec!r}; "
			f"use one of {', '.join(RERANKER_PROVIDERS)}"
		)

	# check model
	if not model:
		raise ValueError(f"Reranker {spec!r} names no model after the provider")

	# return parts
	return provider, model


class Reranker(Protocol):
	"""Orders documents by relevance to a query."""

	name: str
	"""The full spec, for example ``cohere/rerank-v5.0-fast``."""

	def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
		"""Order ``documents`` for ``query``.

		Args:
			query (str): The search query.

			documents (list[str]): The texts to order.

		Returns:
			list[tuple[int, float]]: Every index of ``documents`` once, as
				``(index, score)``, best first.

		Raises:
			RerankError: The documents could not be ordered.
		"""
		...


class CohereReranker:
	"""Reranker backed by the Cohere v2 rerank API."""

	def __init__(
		self,
		model: str,
		api_key: SecretStr,
		timeout: float,
		transport: httpx.BaseTransport | None = None,
	) -> None:
		"""Create the reranker.

		Args:
			model (str): The Cohere model.

			api_key (SecretStr): The Cohere API key.

			timeout (float): Seconds allowed for the network phase.

			transport (httpx.BaseTransport | None): A transport, for testing.
		"""
		self.model = model
		self.name = f"cohere/{model}"

		# set the API key and client
		self._api_key = api_key
		self._client = httpx.Client(
			timeout=httpx.Timeout(timeout, connect=min(timeout, 1.0)),
			transport=transport,
		)

	def __repr__(self) -> str:
		"""Show the name only, never the key."""
		return f"CohereReranker({self.name!r})"

	def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
		"""Order the documents for the query with Cohere.

		Args:
			query (str): The search query.

			documents (list[str]): The texts to order.

		Returns:
			list[tuple[int, float]]: Every document index once, best first.

		Raises:
			RerankError: The call failed or the answer is not a full ranking.
		"""
		# build the request body
		body = {
			"model": self.model,
			"query": query,
			"documents": documents,
			"top_n": len(documents),
		}

		# build the headers
		headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

		# call cohere
		# > httpx encodes the header as ASCII, so a key with another character raises
		# > a UnicodeError that holds the key
		failure = None

		try:
			response = self._client.post(COHERE_RERANK_URL, json=body, headers=headers)

		except (httpx.HTTPError, UnicodeError) as error:
			failure = type(error).__name__

		# raise outside the handler, so no error keeps the header as context
		if failure is not None:
			raise RerankError(f"Cohere rerank failed: {failure}")

		# check status
		if not response.is_success:
			raise RerankError(f"Cohere rerank failed: HTTP {response.status_code}")

		# read results
		try:
			ranked = [
				(int(item["index"]), float(item["relevance_score"]))
				for item in response.json()["results"]
			]

		except (ValueError, KeyError, TypeError):
			raise RerankError("Cohere rerank gave an unexpected answer") from None

		# check every index appears once
		if sorted(index for index, _ in ranked) != list(range(len(documents))):
			raise RerankError("Cohere rerank did not rank every document once")

		# return ranking
		return ranked


class FastembedReranker:
	"""Reranker backed by a local fastembed cross-encoder."""

	_encoder: Any
	"""The fastembed cross-encoder."""

	_lock: threading.Lock
	"""Guards `_encoder`, so two threads load one model."""

	def __init__(self, model: str) -> None:
		"""Create the reranker.

		Args:
			model (str): The fastembed model.
		"""
		self.model = model
		self.name = f"fastembed/{model}"

		# set the encoder and lock
		self._encoder = None
		self._lock = threading.Lock()

	def __repr__(self) -> str:
		"""Show the name only."""
		return f"FastembedReranker({self.name!r})"

	def _loaded_encoder(self) -> Any:
		"""Load the model once, and return it."""
		with self._lock:
			if self._encoder is None:
				self._encoder = _load_cross_encoder(self.model)

			# return!
			return self._encoder

	def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
		"""Order the documents for the query with the cross-encoder.

		Args:
			query (str): The search query.

			documents (list[str]): The texts to order.

		Returns:
			list[tuple[int, float]]: Every document index once, best first.

		Raises:
			RerankError: The model failed to load or to score.
		"""
		# load the encoder
		try:
			encoder = self._loaded_encoder()
			scores = [float(score) for score in encoder.rerank(query, documents)]

		except Exception as error:
			raise RerankError(
				f"fastembed rerank failed: {type(error).__name__}"
			) from error

		# check if the scores are the same length as the documents
		if len(scores) != len(documents):
			raise RerankError("fastembed rerank did not score every document")

		# return the scores sorted by relevance
		return sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)


def check_default_reranker(spec: str) -> None:
	"""Fail unless the default reranker is local or off.

	Args:
		spec (str): The ``default_reranker_model`` setting.

	Raises:
		ValueError: The spec is not valid or names a paid service.
	"""
	# off is allowed
	if not spec:
		return

	provider, _ = parse_reranker_model(spec)

	# check provider
	if provider not in LOCAL_PROVIDERS:
		raise ValueError(
			f"default_reranker_model {spec!r} must be local "
			f"({', '.join(LOCAL_PROVIDERS)}): open use never calls a paid service"
		)


def create_reranker(spec: str, settings: Settings) -> Reranker | None:
	"""Build the reranker a spec names.

	Args:
		spec (str): A reranker setting, ``<provider>/<model>``.

		settings (Settings): The configuration, with the Cohere key and timeout.

	Returns:
		Reranker | None: ``None`` when the spec is off.
	"""
	# off without a model
	if not spec:
		return None

	# parse the reranker model
	provider, model = parse_reranker_model(spec)

	# return the fastembed reranker
	if provider == "fastembed":
		return FastembedReranker(model)

	# return the cohere reranker if it is active
	key = _cohere_key(settings)
	if key is None:
		return None

	# return cohere
	return CohereReranker(model, key, settings.reranker_timeout)


def rerankers_for(settings: Settings, client: str | None) -> list[Reranker]:
	"""Return the rerankers for ``client``.

	Args:
		settings (Settings): The configuration.

		client (str | None): The client, as `access.client_of_request` names it.

	Returns:
		list[Reranker]: The active rerankers.
	"""
	if _is_key_client(client):
		specs = [settings.key_reranker_model, settings.default_reranker_model]

	else:
		specs = py_.filter_([settings.default_reranker_model], _is_local)

	# return!
	return py_.compact([_shared_reranker(spec, settings) for spec in py_.uniq(specs)])


def active_rerankers(settings: Settings) -> dict[str, str | None]:
	"""Name the rerankers the settings turn on.

	Args:
		settings (Settings): The configuration.

	Returns:
		dict[str, str | None]: The ``key_reranker`` and the ``default_reranker``
			spec, each ``None`` when off.
	"""
	# return!
	return {
		"key_reranker": (
			settings.key_reranker_model
			if _is_active(settings.key_reranker_model, settings)
			else None
		),
		"default_reranker": settings.default_reranker_model or None,
	}


def reranker_status(settings: Settings) -> str:
	"""Describe both rerankers for ``serve``.

	Args:
		settings (Settings): The configuration.

	Returns:
		str: The status line.
	"""
	key_status = _status_of_key_reranker(settings)
	default_status = settings.default_reranker_model or "none"

	# check if both are off
	if key_status == default_status == "none":
		return "Reranker: none"

	# return!
	return f"Reranker: {key_status} for keys, {default_status} by default"


def rerank_search(
	engine: Engine,
	settings: Settings,
	rerankers: list[Reranker],
	query: str,
	source: str,
	top_k: int,
) -> tuple[list[dict], str | None]:
	"""Search a source and order the best candidates.

	Args:
		engine (Engine): The database engine.

		settings (Settings): The settings with the search parameters.

		rerankers (list[Reranker]): The rerankers to try.

		query (str): The search query.

		source (str): The corpus tag to search.

		top_k (int): How many hits to return.

	Returns:
		tuple[list[dict], str | None]: The best ``top_k`` chunks, reranked ones
			with ``rerank_score``, and the name of the reranker that ordered
			them, or ``None`` for the fused order.
	"""
	# take the candidates
	hits = store.search_hybrid(
		engine=engine,
		model_name=settings.embedding_model,
		query=query,
		source=source,
		top_k=max(top_k, settings.reranker_candidates) if rerankers else top_k,
		candidate_k=settings.candidate_k,
		rrf_k=settings.rrf_k,
	)

	# return the fused order without candidates
	if not hits:
		return hits, None

	# order the candidates with the first reranker that succeeds
	for reranker in rerankers:
		ranked = _ranking_of(reranker, query, hits)
		if ranked is not None:
			reranked = [
				{**hits[index], "rerank_score": score} for index, score in ranked
			]

			# return the reranked chunks
			return reranked[:top_k], reranker.name

	# return the fused chunks
	return hits[:top_k], None
