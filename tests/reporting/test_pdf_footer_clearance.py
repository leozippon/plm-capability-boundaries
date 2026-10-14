"""Compiled captions must stay above the page-number band."""
import pytest

from scripts.capability.reporting.validate_manuscript import pdf_footer_faults


def geometry(body_bottom=700, footer=True):
    number = '<word xMin="295" xMax="305" yMin="710" yMax="720">1</word>' if footer else ''
    return ('<html><page width="600" height="842">'
            f'<word xMin="50" xMax="150" yMin="690" yMax="{body_bottom}">caption</word>'
            + number + '</page></html>')


def test_caption_clear_of_footer():
    assert pdf_footer_faults(geometry()) == []


def test_caption_reaching_footer_band_is_refused():
    assert pdf_footer_faults(geometry(715)) == ['Body text reaches page-number area on page 1']


def test_unverifiable_footer_is_not_reported_as_clean():
    assert pdf_footer_faults(geometry(footer=False)) == ['Cannot identify unique numbered footer on page 1']
    with pytest.raises(ValueError, match='No pages'):
        pdf_footer_faults('<html/>')
