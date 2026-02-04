import numpy as np
from sklearn.decomposition import PCA


class FeatureExtractor:
    def __init__(
        self,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        n_components=100,
        whiten=True,
        random_state=42
    ):
        # HOG parameters
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        # PCA parameters
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

        self.pca = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state
        )

        self._hog_available = False
        self._check_hog()
        self._is_fitted = False

    # ---------------------------------------------------------
    def _check_hog(self):
        try:
            from skimage.feature import hog  # noqa
            self._hog_available = True
        except Exception:
            self._hog_available = False

    # ---------------------------------------------------------
    def _extract_hog(self, img):
        from skimage.feature import hog

        return hog(
            img,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=False,
            feature_vector=True
        )

    # ---------------------------------------------------------
    def _fallback_features(self, img):
        grad_x = np.diff(img, axis=1)
        grad_y = np.diff(img, axis=0)
        return np.array([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(grad_x),
            np.std(grad_y),
            np.mean(img),
            np.std(img)
        ])

    # ---------------------------------------------------------
    def _extract_features_single(self, img):
        img = np.asarray(img)

        if img.ndim == 1:
            img = img.reshape(28, 28)

        if self._hog_available:
            return self._extract_hog(img)
        else:
            return self._fallback_features(img)

    # ---------------------------------------------------------
    def _extract_hog_batch(self, images):
        feats = []
        for img in images:
            feats.append(self._extract_features_single(img))
        return np.array(feats)

    # ---------------------------------------------------------
    def fit(self, images):
        """
        Fit PCA on HOG features (TRAIN ONLY)
        """
        hog_feats = self._extract_hog_batch(images)
        self.pca.fit(hog_feats)
        self._is_fitted = True
        return self

    # ---------------------------------------------------------
    def transform(self, images):
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureExtractor is not fitted. "
                "Run training before testing."
            )

        hog_feats = self._extract_hog_batch(images)
        return self.pca.transform(hog_feats)

    # ---------------------------------------------------------
    def fit_transform(self, images):
        hog_feats = self._extract_hog_batch(images)
        feats = self.pca.fit_transform(hog_feats)
        self._is_fitted = True
        return feats

    # ---------------------------------------------------------
    def extract_features_batch(self, images):
        """
        Compatibility with test code
        """
        return self.transform(images)

    # ---------------------------------------------------------
    def extract_all_features(self, images):
        feats = self.transform(images)
        info = {"features": feats.shape[1]}
        return feats, info
