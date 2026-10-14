"""Named manuscript roles from author-supplied publication palette anchors.

APOLLO binary blue/orange: https://www.nature.com/articles/s43588-025-00948-w
FedProt Text/Protein/Joint/Adapted: https://www.nature.com/articles/s43588-025-00832-7
Domainome negative/zero/positive: https://www.nature.com/articles/s41586-024-08370-4
Anchors are author-supplied, not sampled or inferred from publication images.
White/slate neutrals follow the supplied editorial reference PNG; light semantic
box tints are derived from these anchors. These roles do not reclassify cohorts.
"""
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

PATH = Path(__file__).resolve()
BINARY_FIRST, BINARY_SECOND = "#44729D", "#D48640"
MODEL_CLASSES = ("#4575B4", "#7BCCC4", "#E6B400", "#C62921")
NEGATIVE, ZERO, POSITIVE = "#ED2122", "#FFFFFF", "#9F8CC3"
CONTROL = "#000000"
WHITE = ZERO
INK, SLATE, BORDER, NEUTRAL = "#262C33", "#44546B", "#D6DEE5", "#E7EDF1"
BACKBONE, SITE, SECOND_SITE = BINARY_FIRST, NEGATIVE, BINARY_SECOND
NATIVE, READOUT = BINARY_FIRST, BINARY_SECOND
ANY_FAMILY, COMPLETE_DOMAIN = BINARY_FIRST, BINARY_SECOND
DOWNSTREAM = BACKBONE
JOINT_TERM, TERM_CONTRAST = MODEL_CLASSES[1], MODEL_CLASSES[2]
SIGNED = LinearSegmentedColormap.from_list("domainome_signed", [NEGATIVE, ZERO, POSITIVE], N=257)


def tint(color: str, fraction: float = .12) -> str:
    """Small semantic boxes only: mix an anchor with a white background."""
    red, green, blue = to_rgb(color)
    return to_hex((1 - fraction + fraction * red, 1 - fraction + fraction * green,
                   1 - fraction + fraction * blue))
