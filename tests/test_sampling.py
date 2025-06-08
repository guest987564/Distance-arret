import numpy as np
from sampling import sample_tr, tr_pdf


def test_sample_tr_bounds():
    rng = np.random.default_rng(0)
    data = sample_tr("Standard", 1000, rng)
    assert (data >= 0.3).all() and (data <= 3).all()


def test_tr_pdf_normalization():
    xs = np.linspace(0.3, 3, 100)
    pdf = tr_pdf(xs, "Standard")
    area = np.trapz(pdf, xs)
    assert abs(area - 1) < 0.05
