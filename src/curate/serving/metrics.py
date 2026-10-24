"""Prometheus metrics with a registry per application instance.

A fresh ``CollectorRegistry`` per app avoids duplicate-registration errors when tests build several apps
in one process, and keeps the metrics of two bundles apart if they ever share a process.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


@dataclass(frozen=True)
class Metrics:
    """All collectors for one app."""

    registry: CollectorRegistry
    requests_total: Counter
    request_duration: Histogram
    recommendations_total: Counter
    model_latency: Histogram
    cold_starts_total: Counter
    info: Gauge

    @classmethod
    def create(cls) -> Metrics:
        """Build the collectors on a new registry."""
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests_total=Counter(
                "curate_http_requests_total",
                "HTTP requests",
                ["method", "route", "status"],
                registry=registry,
            ),
            request_duration=Histogram(
                "curate_http_request_duration_seconds",
                "HTTP request latency",
                ["method", "route"],
                buckets=LATENCY_BUCKETS,
                registry=registry,
            ),
            recommendations_total=Counter(
                "curate_recommendations_total", "Recommendation requests served", ["model"], registry=registry
            ),
            model_latency=Histogram(
                "curate_model_latency_seconds",
                "Time spent scoring and ranking inside the model",
                ["model"],
                buckets=LATENCY_BUCKETS,
                registry=registry,
            ),
            cold_starts_total=Counter(
                "curate_cold_start_total",
                "Requests for unknown users served by the popularity fallback",
                registry=registry,
            ),
            info=Gauge("curate_bundle_info", "Loaded bundle", ["dataset", "version"], registry=registry),
        )
