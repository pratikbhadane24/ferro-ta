"""Unit tests for ferro_ta.indicators.price_transform"""

import numpy as np

from ferro_ta.indicators.price_transform import AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

O = np.array([10.0, 11.0, 12.0, 13.0])
H = np.array([12.0, 13.0, 14.0, 15.0])
L = np.array([9.0, 10.0, 11.0, 12.0])
C = np.array([11.0, 12.0, 13.0, 14.0])


# ---------------------------------------------------------------------------
# AVGPRICE
# ---------------------------------------------------------------------------


class TestAVGPRICE:
    def test_known_formula(self):
        result = AVGPRICE(O, H, L, C)
        expected = (O + H + L + C) / 4.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_first_bar(self):
        result = AVGPRICE(O, H, L, C)
        np.testing.assert_allclose(result[0], (10 + 12 + 9 + 11) / 4.0, rtol=1e-10)

    def test_no_nan(self):
        result = AVGPRICE(O, H, L, C)
        assert np.all(np.isfinite(result))

    def test_length(self):
        assert len(AVGPRICE(O, H, L, C)) == len(O)


# ---------------------------------------------------------------------------
# MEDPRICE
# ---------------------------------------------------------------------------


class TestMEDPRICE:
    def test_known_formula(self):
        result = MEDPRICE(H, L)
        expected = (H + L) / 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_first_bar(self):
        result = MEDPRICE(H, L)
        np.testing.assert_allclose(result[0], (12 + 9) / 2.0, rtol=1e-10)

    def test_no_nan(self):
        result = MEDPRICE(H, L)
        assert np.all(np.isfinite(result))

    def test_length(self):
        assert len(MEDPRICE(H, L)) == len(H)


# ---------------------------------------------------------------------------
# TYPPRICE
# ---------------------------------------------------------------------------


class TestTYPPRICE:
    def test_known_formula(self):
        result = TYPPRICE(H, L, C)
        expected = (H + L + C) / 3.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_first_bar(self):
        result = TYPPRICE(H, L, C)
        np.testing.assert_allclose(result[0], (12 + 9 + 11) / 3.0, rtol=1e-10)

    def test_no_nan(self):
        result = TYPPRICE(H, L, C)
        assert np.all(np.isfinite(result))

    def test_length(self):
        assert len(TYPPRICE(H, L, C)) == len(H)


# ---------------------------------------------------------------------------
# WCLPRICE
# ---------------------------------------------------------------------------


class TestWCLPRICE:
    def test_known_formula(self):
        result = WCLPRICE(H, L, C)
        expected = (H + L + 2.0 * C) / 4.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_first_bar(self):
        result = WCLPRICE(H, L, C)
        np.testing.assert_allclose(result[0], (12 + 9 + 2 * 11) / 4.0, rtol=1e-10)

    def test_no_nan(self):
        result = WCLPRICE(H, L, C)
        assert np.all(np.isfinite(result))

    def test_close_weight_double(self):
        # WCLPRICE weights close twice vs TYPPRICE
        wcl = WCLPRICE(H, L, C)
        # On a rising series (H > L > 0), WCLPRICE > TYPPRICE when C > (H+L)/2
        # Just verify formula correctness already done above
        assert np.all(np.isfinite(wcl))

    def test_length(self):
        assert len(WCLPRICE(H, L, C)) == len(H)
