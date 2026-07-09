"""Cycle-8 declarative-serving infrastructure (WP-C8).

The Topaz-keyed capability registry and the operator-curated composition-shape
catalog (ADR-047). Per AS-11 the registry is not a parallel structure: parts are
the existing capability-ensemble library keyed by Topaz skill, shapes are
operator-authored ensemble skeletons declaring the intent they serve, and both
are admitted through the single AS-2 routine (:mod:`admission`).
"""
