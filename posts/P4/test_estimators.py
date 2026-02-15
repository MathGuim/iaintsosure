import pytest

from sklearn.utils.estimator_checks import parametrize_with_checks

from custom_estimators import (
    FFTRepresentation,
    PCAOutlierDetector,
    PCAMahalanobisOutlierDetector
)

@parametrize_with_checks([FFTRepresentation(n_components=1), PCAOutlierDetector(), PCAMahalanobisOutlierDetector()])
def test_estimator_compliance(estimator, check):
    check(estimator)

