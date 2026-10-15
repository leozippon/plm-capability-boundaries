"""Current publication locations; historical figure and source-table IDs stay fixed."""
from pathlib import Path

PATH = Path(__file__).resolve()
SUBMISSION = Path('manuscript/submission')
PDF_DIRECTORY = SUBMISSION / 'figures'
ASSETS = Path('manuscript/assets')
FRAMEWORK_SOURCE = ASSETS / 'framework-original.png'
STRUCTURE_SELECTION = ASSETS / 'selection.json'
CHECKPOINT_KEY = SUBMISSION / 'checkpoint-name-key.tex'

PDF_NAMES = {
    'Fig1_framework': 'figure-1-framework.pdf',
    'Fig2_mutation': 'figure-2-mutation.pdf',
    'Fig3_position': 'figure-3-position.pdf',
    'Fig4_phenotypes_and_interactions': 'figure-4-phenotypes-and-interactions.pdf',
    'Fig5_context_and_generation': 'figure-5-context-and-generation.pdf',
    'Fig6_guidance_and_selection': 'figure-6-guidance-and-selection.pdf',
    'native-comparators': 'supplementary-figure-1-native-comparators.pdf',
    'conditional-class-yield': 'supplementary-figure-2-conditional-class-yield.pdf',
    'generation-yield': 'supplementary-figure-3-generation-yield.pdf',
    'matched-generators': 'supplementary-figure-4-matched-generators.pdf',
}


def figure_pdf(figure: str) -> Path:
    """Repository-relative PDF destination; unknown figure IDs are an error."""
    return PDF_DIRECTORY / PDF_NAMES[figure]
