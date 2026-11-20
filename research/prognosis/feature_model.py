"""Forward của MultimodalBrainTumorModel từ đặc trưng DenseNet121 đã tính sẵn (backbone đóng băng).

Code nằm ở backend/ai_core/architectures/feature_forward.py để serving dùng đúng cùng một bản; file này chỉ re-export.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from ai_core.architectures.feature_forward import forward_from_features, pooled_backbone_features  # noqa: E402,F401
