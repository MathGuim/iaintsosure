import numpy as np

from scipy import fftpack

from numbers import Real, Integral

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from sklearn.base import (
    BaseEstimator,
    OutlierMixin,
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin
)
from sklearn.utils.validation import check_is_fitted, check_scalar, validate_data


class FFTRepresentation(TransformerMixin, ClassNamePrefixFeaturesOutMixin, BaseEstimator):
    """
    Transformer to create a feature representation using the inverse Fast Fourier
    Transform (IFFT) of the first 's' coefficients.

    The representation length is determined by 's'.

    Parameters
    ----------
    s : int or None, default=None
        The number of initial FFT coefficients to keep.
        If None, the number of coefficients kept will be equal to the
        number of timesteps (X.shape[1]).
    """

    def __init__(self, *, n_components: int):
        self.n_components = n_components
        self._n_components = n_components

    def fit(self, X, y=None):
        """
        Fits the transformer by determining the number of features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The training input samples (time series).
        y : None
            Ignored. Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns self.
        """

        if self._n_components is not None:
            self._n_components = check_scalar(self._n_components, "s", Integral, min_val=1)
        else:
            self._n_components = X.shape[1]

        X = validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype="numeric",
            reset=True
        )
        
        if self._n_components > X.shape[1]:
             raise ValueError(
                f"Parameter 's' ({self._n_components}) cannot be greater than the "
                f"number of timesteps in X n_features={X.shape[1]}."
            )
        
        self.n_features_in_ = X.shape[1]
        self._n_features_out = self._n_components

        return self
    
    def transform(self, X):
        """
        Applies the FFT-based transformation to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The input samples (time series).

        Returns
        -------
        X_new : ndarray of shape (n_samples, s)
            The transformed data, where 's' is the number of features/coefficients
            kept. The output is the real part of the IFFT result.
        """
        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype="numeric",
            reset=False
        )
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this transformer "
                f"is fitted with {self.n_features_in_} features."
            )
        
        n_samples = X.shape[0]

        X_new = np.empty((n_samples, self._n_components))

        # Process each sample separately
        for i in range(n_samples):
            # Compute FFT and keep only the first self._n_components coefficients
            fft_coef = fftpack.fft(X[i])[:self._n_components]
            representation = fftpack.ifft(fft_coef).real
            X_new[i, :] = representation

        return X_new
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        return tags


class PCAOutlierDetector(OutlierMixin, BaseEstimator):
    """
    Outlier detector based on Principal Component Analysis (PCA) reconstruction error.
    """

    def __init__(self, contamination=0.1, n_components=2, random_state=42):
        self.contamination = contamination
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        self.contamination = check_scalar(
            self.contamination, "contamination", Real, min_val=0.0, max_val=0.5, 
            include_boundaries="right"
        )
        
        X = validate_data(self, X=X, accept_sparse=False, ensure_min_samples=2, dtype=np.float64)

        self.n_features_in_ = X.shape[1]

        self.pipeline_ = make_pipeline(
            StandardScaler(),
            PCA(n_components=self.n_components, random_state=self.random_state)
        )
        self.pipeline_.fit(X)

        self.components_ = self.pipeline_.transform(X)
        reconstructed = self.pipeline_.inverse_transform(self.components_)
        
        # Determine error scores (always positive)
        self.scores_ = np.linalg.norm(X - reconstructed, axis=1)

        self.threshold_ = np.quantile(self.scores_, 1-self.contamination)

        # Define offset_
        # Since score_samples = -error, and decision = threshold - error
        # Then: threshold - error = -error - offset_  =>  offset_ = -threshold
        self.offset_ = -self.threshold_

        self.labels_ = self.predict(X)

        return self

    def decision_function(self, X):
        """
        Returns the shifted anomaly score.
        Positive = Inlier. Negative = Outlier.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=False, ensure_2d=True, dtype=np.float64, reset=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature mismatch: {X.shape[1]} vs {self.n_features_in_}")

        compressed = self.pipeline_.transform(X)
        reconstructed = self.pipeline_.inverse_transform(compressed)
        reconstruction_error = np.linalg.norm(X - reconstructed, axis=1)

        # Shift result so 0 is the threshold
        # Positive values (small error) -> Inliers
        # Negative values (large error) -> Outliers
        return self.threshold_ - reconstruction_error

    def score_samples(self, X):
        """
        Returns the negative reconstruction error.
        Higher value = Better (Lower error).
        """
        # This calls decision_function and reverses the offset math
        # score = decision + offset
        # score = (threshold - error) + (-threshold) = -error
        return self.decision_function(X) + self.offset_

    def predict(self, X):
        """
        Predict -1 for outliers and 1 for inliers.
        """
        check_is_fitted(self)
        
        # Use decision_function sign for prediction
        # decision >= 0 means Inlier (1)
        # decision < 0  means Outlier (-1)
        decision = self.decision_function(X)
        is_inlier = decision >= 0
        
        return np.where(is_inlier, 1, -1)
    

class PCAMahalanobisOutlierDetector(OutlierMixin, BaseEstimator):
    """
    Outlier detector using PCA for dimensionality reduction followed by
    Mahalanobis distance calculation.
    """

    def __init__(self, contamination=0.1, n_components=2, random_state=42):
        self.contamination = contamination
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        """
        #  Use self.contamination
        self.contamination = check_scalar(
            self.contamination, "contamination", Real, min_val=0.0, max_val=0.5, 
            include_boundaries="right"
        )

        # Use validate_data and force float64
        X = validate_data(self, X=X, accept_sparse=False, ensure_min_samples=2, dtype=np.float64)
        
        self.n_features_in_ = X.shape[1]

        self.pipeline_ = make_pipeline(
            StandardScaler(),
            PCA(n_components=self.n_components, random_state=self.random_state)
        )

        self.components_ = self.pipeline_.fit_transform(X)
        if self.components_.ndim == 1:
            self.components_ = self.components_.reshape(-1, 1) # (n_samples, 1)

        self.mean_ = self.components_.mean(axis=0)
        
        # Ensure covariance is at least 2D (e.g., (1, 1) matrix)
        self.cov_ = np.cov(self.components_, rowvar=False)
        if self.cov_.ndim == 0:
             self.cov_ = self.cov_.reshape((1, 1)) # Handle scalar case

        # Regularize and Invert
        eps = 1e-6
        self.cov_ += eps * np.eye(self.cov_.shape[0])
        self.inv_cov_ = np.linalg.inv(self.cov_)
        
        # Calculate Mahalanobis distance on training data
        diff = self.components_ - self.mean_
        self.scores_ = np.sqrt(np.sum(diff @ self.inv_cov_ * diff, axis=1))

        n_samples = self.scores_.shape[0]
        n_outliers = int(n_samples * self.contamination)
        sorted_scores = np.sort(self.scores_)
        
        if n_outliers == 0:
            # If no outliers expected, threshold is higher than max score
            self.threshold_ = sorted_scores[-1] + 1e-10
        else:
            # The threshold is the score of the lowest-scoring outlier
            # (Top K scores are outliers)
            cutoff_index = n_samples - n_outliers 
            self.threshold_ = sorted_scores[cutoff_index - 1]

        self.offset_ = -self.threshold_
        self.labels_ = self.predict(X)
        
        return self

    def decision_function(self, X):
        """
        Compute the shifted anomaly score.
        Positive = Inlier. Negative = Outlier.
        """
        check_is_fitted(self)

        X = validate_data(self, X=X, accept_sparse=False, dtype=np.float64, reset=False)
    
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this PCAMahalanobisOutlierDetector "
                f"is fitted with {self.n_features_in_} features."
            )

        components = self.pipeline_.transform(X)
        diff = components - self.mean_
        
        # Raw Mahalanobis distance
        mahal_dist = np.sqrt(np.sum(diff @ self.inv_cov_ * diff, axis=1))

        # If Distance > Threshold (Outlier) -> Result is Negative
        # If Distance < Threshold (Inlier)  -> Result is Positive
        return self.threshold_ - mahal_dist

    def score_samples(self, X):
        """
        Compute the negative Mahalanobis distance.
        Higher value = Better (closer to center).
        """
        # score = decision + offset
        # score = (threshold - dist) + (-threshold) = -dist
        return self.decision_function(X) + self.offset_

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        """
        decision = self.decision_function(X)
        return np.where(decision >= 0, 1, -1)


