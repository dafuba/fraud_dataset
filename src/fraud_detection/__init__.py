"""Fraud detection toolkit.

Modules
-------
config      project paths, column groups, seed
features    leakage-safe feature engineering (BasicFeatures, UserProfileEncoder)
pipeline    sklearn pipelines and candidate models
evaluation  cross-validated model comparison and hold-out reports
costs       business cost model, threshold optimisation, three-way decisions
explain     SHAP-based reason codes for analysts
drift       population stability index (PSI) monitoring
training    reproducible training + export of deployable artefacts
viz         consistent, colour-blind-safe plotting defaults
"""

__version__ = "1.0.0"
