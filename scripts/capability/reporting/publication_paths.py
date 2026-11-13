"""Current publication locations; historical figure and source-table IDs stay fixed."""
from pathlib import Path

PATH = Path(__file__).resolve()
SUBMISSION = Path('manuscript/submission')
PDF_DIRECTORY = SUBMISSION / 'figures'
ASSETS = Path('manuscript/assets')
FRAMEWORK_SOURCE = ASSETS / 'framework-original.png'
STRUCTURE_SELECTION = ASSETS / 'selection.json'
CHECKPOINT_KEY = SUBMISSION / 'checkpoint-name-key.tex'

#: Figures this repository renders into the submission, each mapped to the
#: filename the manuscript includes it under. Figure 1 is the only one: every
#: other submitted figure is an author export inserted byte for byte from the
#: Figma masters the package manifest records, under a numbering nothing here
#: produces. The nine entries that named the superseded 2026-10-15 set are
#: retired, because not one of those filenames appears in the manuscript. The
#: historical IDs themselves stay fixed where they are still carried: in the
#: retained source and plotted tables under ``figure_data/``, which are the
#: distributed form of the manuscript's evidence.
PDF_NAMES = {
    'Fig1_framework': 'figure-1-framework.pdf',
}


def figure_pdf(figure: str) -> Path:
    """Repository-relative PDF destination; unknown figure IDs are an error."""
    return PDF_DIRECTORY / PDF_NAMES[figure]
