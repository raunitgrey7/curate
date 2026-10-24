"""curate: a recommender platform with offline, counterfactual and sequential A/B evaluation.

The package is organised by concern: ``data`` loads MovieLens 100k and generates a seeded retail
catalogue, ``models`` holds five recommenders behind one protocol, ``features`` is a sqlite feature
store, ``evaluation`` implements ranking metrics, IPS/SNIPS estimators and an A/B simulator, and
``serving`` exposes everything over FastAPI with explanations.
"""

__version__ = "0.1.0"
