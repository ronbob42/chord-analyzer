"""Chord template definitions for template-based chord recognition."""

import numpy as np

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Binary chord templates (root = C, index 0)
# Each template encodes which pitch classes are present in the chord.
CHORD_TYPES = {
    'maj':  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # root, M3, P5
    'min':  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # root, m3, P5
    '5':    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # root, P5 (power chord)
    'dim':  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # root, m3, dim5
    'aug':  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # root, M3, aug5
    '7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # root, M3, P5, m7
    'maj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # root, M3, P5, M7
    'min7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # root, m3, P5, m7
}


def generate_all_templates() -> dict[str, np.ndarray]:
    """Generate chord templates for all 12 roots and all chord types.

    Returns dict mapping chord name (e.g. 'C maj', 'F# min7') to a
    12-dimensional numpy array. Also includes 'N' (no chord).
    Total: 12 roots * 8 types + 1 = 97 templates.
    """
    templates: dict[str, np.ndarray] = {}
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for chord_type, base_template in CHORD_TYPES.items():
            rotated = np.roll(base_template, root_idx)
            chord_name = f"{root_name} {chord_type}"
            templates[chord_name] = rotated.astype(float)
    templates['N'] = np.zeros(12)
    return templates
