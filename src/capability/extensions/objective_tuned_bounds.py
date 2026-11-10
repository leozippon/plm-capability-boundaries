"""The two instrument bounds the third amendment of ``docs/objective-alignment.md`` carries.

Kept apart from :mod:`.objective_tuned_prediction` because the correlation-unit floor that
declaration freezes is *measured from* these bounds, so the measurement cannot import the
declaration that waits on it.
"""

#: The coupling fitter's run-to-run spread of one per-variant Potts score on a deep
#: background, in score units.
COUPLING_RUN_SPREAD = 1.07e-2

#: A fit stopped at its iteration cap moves a per-variant score about this many times the
#: run-to-run spread, which is why a non-converged background is booked incomplete and
#: never used.
CAPPED_FIT_FACTOR = 4
