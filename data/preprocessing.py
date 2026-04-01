"""
SAR Preprocessing for Flood Detection

Techniques: histogram matching, z-score, adaptive percentile, log-ratio, NDI, CLAHE
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any


class SARPreprocessor:
    """
    Advanced SAR preprocessing for bi-temporal flood detection.

    This preprocessor addresses common SAR data challenges:
    - Varying acquisition conditions (orbit, angle, weather)
    - Speckle noise
    - Subtle flood signals
    - Class overlap in feature space

    Usage:
        preprocessor = SARPreprocessor(
            use_histogram_matching=True,
            use_log_ratio=True
        )
        pre_processed, post_processed = preprocessor(pre, post)

    Attributes:
        use_histogram_matching: Match post histogram to pre (fixes acquisition bias)
        use_zscore: Per-image z-score standardization
        use_adaptive_norm: Percentile-based normalization
        use_log_ratio: Add log-ratio change channel
        use_ndi: Add normalized difference index channel
        use_clahe: Apply contrast-limited adaptive histogram equalization
    """

    def __init__(
        self,
        use_histogram_matching: bool = False,
        use_zscore: bool = False,
        use_adaptive_norm: bool = False,
        use_log_ratio: bool = False,
        use_ndi: bool = False,
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        adaptive_percentile_low: float = 2.0,
        adaptive_percentile_high: float = 98.0,
        log_ratio_clip: float = 3.0,
        eps: float = 1e-6
    ):
        self.use_histogram_matching = use_histogram_matching
        self.use_zscore = use_zscore
        self.use_adaptive_norm = use_adaptive_norm
        self.use_log_ratio = use_log_ratio
        self.use_ndi = use_ndi
        self.use_clahe = use_clahe

        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.adaptive_percentile_low = adaptive_percentile_low
        self.adaptive_percentile_high = adaptive_percentile_high
        self.log_ratio_clip = log_ratio_clip
        self.eps = eps

        # Try to import optional dependencies
        self._cv2 = None
        self._skimage_exposure = None

        if use_clahe:
            try:
                import cv2  # noqa: F401
            except ImportError:
                raise ImportError("OpenCV (cv2) required for CLAHE. Install with: pip install opencv-python")

        if use_histogram_matching:
            try:
                from skimage import exposure  # noqa: F401
            except ImportError:
                raise ImportError("scikit-image required for histogram matching. Install with: pip install scikit-image")

    def __call__(
        self,
        pre: np.ndarray,
        post: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Apply preprocessing to pre and post event images.

        Args:
            pre: Pre-event image, shape (H, W) or (C, H, W)
            post: Post-event image, shape (H, W) or (C, H, W)

        Returns:
            Tuple of:
                - pre_processed: Preprocessed pre-event image
                - post_processed: Preprocessed post-event image
                - extra_channels: Dict of additional computed channels (log_ratio, ndi, etc.)
        """
        # Handle both 2D and 3D inputs
        squeeze_output = False
        if pre.ndim == 2:
            pre = pre[np.newaxis, ...]  # (1, H, W)
            post = post[np.newaxis, ...]
            squeeze_output = True

        extra_channels = {}

        # Step 1: Acquisition normalization (per-channel)
        pre_norm = pre.copy()
        post_norm = post.copy()

        for c in range(pre.shape[0]):
            pre_ch = pre[c]
            post_ch = post[c]

            # Histogram matching (match post to pre)
            if self.use_histogram_matching:
                post_ch = self._histogram_match(post_ch, pre_ch)

            # CLAHE
            if self.use_clahe:
                pre_ch = self._apply_clahe(pre_ch)
                post_ch = self._apply_clahe(post_ch)

            # Adaptive percentile normalization
            if self.use_adaptive_norm:
                pre_ch = self._adaptive_normalize(pre_ch)
                post_ch = self._adaptive_normalize(post_ch)

            # Z-score normalization
            if self.use_zscore:
                pre_ch = self._zscore_normalize(pre_ch)
                post_ch = self._zscore_normalize(post_ch)

            pre_norm[c] = pre_ch
            post_norm[c] = post_ch

        # Step 2: Compute change features
        if self.use_log_ratio:
            log_ratio = self._compute_log_ratio(pre_norm, post_norm)
            extra_channels['log_ratio'] = log_ratio

        if self.use_ndi:
            ndi = self._compute_ndi(pre_norm, post_norm)
            extra_channels['ndi'] = ndi

        # Compute standard difference (always useful)
        diff = post_norm - pre_norm
        extra_channels['difference'] = diff

        if squeeze_output:
            pre_norm = pre_norm[0]
            post_norm = post_norm[0]
            extra_channels = {k: v[0] if v.ndim > 2 else v for k, v in extra_channels.items()}

        return pre_norm, post_norm, extra_channels

    def _histogram_match(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if self._skimage_exposure is None:
            from skimage import exposure
            self._skimage_exposure = exposure

        return self._skimage_exposure.match_histograms(source, reference)

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2

        # Convert to uint8 for CLAHE
        img_uint8 = (img * 255).astype(np.uint8)

        clahe = self._cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )

        enhanced = clahe.apply(img_uint8)

        return enhanced.astype(np.float32) / 255.0

    def _adaptive_normalize(self, img: np.ndarray) -> np.ndarray:
        p_low = np.percentile(img, self.adaptive_percentile_low)
        p_high = np.percentile(img, self.adaptive_percentile_high)

        return np.clip((img - p_low) / (p_high - p_low + self.eps), 0, 1)

    def _zscore_normalize(self, img: np.ndarray) -> np.ndarray:
        mean = img.mean()
        std = img.std()

        return (img - mean) / (std + self.eps)

    def _compute_log_ratio(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        # Ensure positive values for log
        pre_safe = np.maximum(pre, self.eps)
        post_safe = np.maximum(post, self.eps)

        log_ratio = np.log(post_safe / pre_safe)

        # Clip to prevent extreme values
        log_ratio = np.clip(log_ratio, -self.log_ratio_clip, self.log_ratio_clip)

        # Normalize to approximately [-1, 1]
        log_ratio = log_ratio / self.log_ratio_clip

        return log_ratio.astype(np.float32)

    def _compute_ndi(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        numerator = post - pre
        denominator = post + pre + self.eps

        ndi = numerator / denominator

        return ndi.astype(np.float32)


def create_preprocessor_from_config(config: Dict[str, Any]) -> SARPreprocessor:
    return SARPreprocessor(
        use_histogram_matching=config.get('use_histogram_matching', False),
        use_zscore=config.get('use_zscore', False),
        use_adaptive_norm=config.get('use_adaptive_norm', False),
        use_log_ratio=config.get('use_log_ratio', False),
        use_ndi=config.get('use_ndi', False),
        use_clahe=config.get('use_clahe', False),
        clahe_clip_limit=config.get('clahe_clip_limit', 2.0),
        clahe_tile_size=config.get('clahe_tile_size', 8),
        adaptive_percentile_low=config.get('adaptive_percentile_low', 2.0),
        adaptive_percentile_high=config.get('adaptive_percentile_high', 98.0),
        log_ratio_clip=config.get('log_ratio_clip', 3.0),
        eps=config.get('eps', 1e-6)
    )

# Preset configurations for each dataset
PRESET_CONFIGS = {
    'sen1floods11': {
        # Sen1Floods11: Subtle distribution shift, needs change amplification
        'use_histogram_matching': False,  # Not needed, acquisitions are consistent
        'use_zscore': False,
        'use_adaptive_norm': True,   # Handles varying SAR conditions
        'use_log_ratio': True,       # Amplifies subtle flood signal
        'use_ndi': False,
        'use_clahe': False,
        'log_ratio_clip': 3.0,
    },
    's1gfloods': {
        # S1GFloods: Already excellent separation, minimal processing
        'use_histogram_matching': False,
        'use_zscore': False,
        'use_adaptive_norm': False,  # 8-bit data already normalized
        'use_log_ratio': False,      # Not needed, signal is strong
        'use_ndi': False,
        'use_clahe': False,
    },
    'ombrias1': {
        # OmbriaS1: Heavy class overlap, needs acquisition normalization
        'use_histogram_matching': True,   # Critical! Fixes acquisition bias
        'use_zscore': True,               # Further standardization
        'use_adaptive_norm': False,
        'use_log_ratio': True,            # Amplify change signal
        'use_ndi': False,
        'use_clahe': False,
        'log_ratio_clip': 3.0,
    }
}

def get_preset_preprocessor(dataset_name: str) -> SARPreprocessor:
    dataset_name = dataset_name.lower()

    if dataset_name not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available presets: {list(PRESET_CONFIGS.keys())}"
        )

    return create_preprocessor_from_config(PRESET_CONFIGS[dataset_name])
