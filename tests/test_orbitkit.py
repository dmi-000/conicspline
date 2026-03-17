"""Test suite for orbitkit package."""

import numpy as np
import pytest
from orbitkit import (
    fit_conic, classify_conic, conic_foci, fit_orbit,
    find_orbit_from_points, find_3d_foci, validate_points,
    orbit_type, plane_from_normal, orbit_to_3d,
    OrbitalElements, conic_to_elements, elements_to_conic,
    elements_to_state, orbit_curve, nu_to_time, time_to_pos,
    propagate_orbit, orbital_velocity, propagate_moving_focus,
)


# ── Circular orbit (r=1 AU, T=1 year, mu=4*pi^2) ────────────────────

MU_SOLAR = 4 * np.pi**2  # AU^3/yr^2


class TestCircularOrbit:
    def setup_method(self):
        self.a = 1.0  # AU
        self.e = 0.0
        self.p = 1.0
        self.elem = OrbitalElements(a=1.0, e=0.0, inc=0.0, Omega=0.0,
                                     omega=0.0, nu=0.0)

    def test_period(self):
        # T = 2*pi*sqrt(a^3/mu) = 2*pi*sqrt(1/(4*pi^2)) = 1 year
        T = 2 * np.pi * np.sqrt(self.a**3 / MU_SOLAR)
        assert abs(T - 1.0) < 1e-12

    def test_conic_classification(self):
        # Generate 8 equally-spaced points on a unit circle
        theta = np.linspace(0, 2 * np.pi, 9)[:-1]
        pts = np.column_stack([np.cos(theta), np.sin(theta)])
        coeffs = fit_conic(pts)
        ctype = classify_conic(coeffs)
        assert ctype == "ellipse"

    def test_propagate_cardinal_points(self):
        T = 1.0  # year
        times = np.array([0, T / 4, T / 2, 3 * T / 4])
        pos = propagate_orbit(self.elem, times, mu=MU_SOLAR)
        # In xy-plane: (1,0,0), (0,1,0), (-1,0,0), (0,-1,0)
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=float)
        np.testing.assert_allclose(pos, expected, atol=1e-10)

    def test_velocity_constant_magnitude(self):
        # v = sqrt(mu/a) for circular orbit
        v_expected = np.sqrt(MU_SOLAR / self.a)
        for nu in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            v = orbital_velocity(self.elem, nu, mu=MU_SOLAR)
            assert abs(np.linalg.norm(v) - v_expected) < 1e-10


# ── Elliptical orbit (e=0.5) ─────────────────────────────────────────

class TestEllipticalOrbit:
    def setup_method(self):
        self.a = 2.0  # AU
        self.e = 0.5
        self.p = self.a * (1 - self.e**2)  # = 1.5
        self.elem = OrbitalElements(a=2.0, e=0.5, inc=0.0, Omega=0.0,
                                     omega=0.0, nu=0.0)

    def test_semi_latus_rectum(self):
        assert abs(self.elem.p - 1.5) < 1e-12

    def test_fit_recovers_eccentricity(self):
        # Generate 10 points on the orbit
        nu = np.linspace(0, 2 * np.pi, 11)[:-1]
        r = self.p / (1 + self.e * np.cos(nu))
        pts = np.column_stack([r * np.cos(nu), r * np.sin(nu)])
        # Focus is at origin
        p_fit, e_fit, _ = fit_orbit(pts)
        assert abs(e_fit - 0.5) < 1e-10
        assert abs(p_fit - 1.5) < 1e-10

    def test_conic_to_elements_roundtrip(self):
        elem = conic_to_elements(self.p, self.e, omega=0.3,
                                  inc=0.5, Omega=1.2, nu=0.7)
        p2, e2, w2, i2, O2 = elements_to_conic(elem)
        assert abs(p2 - self.p) < 1e-12
        assert abs(e2 - self.e) < 1e-12
        assert abs(w2 - 0.3) < 1e-12
        assert abs(i2 - 0.5) < 1e-12
        assert abs(O2 - 1.2) < 1e-12

    def test_state_at_periapsis(self):
        # At nu=0: r = a(1-e) = 1 AU
        pos, vel = elements_to_state(self.elem, mu=MU_SOLAR)
        r = np.linalg.norm(pos)
        assert abs(r - self.a * (1 - self.e)) < 1e-10
        # vis-viva: v^2 = mu*(2/r - 1/a)
        v_expected = np.sqrt(MU_SOLAR * (2 / r - 1 / self.a))
        assert abs(np.linalg.norm(vel) - v_expected) < 1e-10

    def test_propagate_roundtrip(self):
        # Generate positions starting from nu=0 (periapsis)
        nu_vals = np.linspace(0, np.pi - 0.1, 6)
        times = np.array([float(nu_to_time(v, self.e, self.p, MU_SOLAR))
                          for v in nu_vals])
        # elem.nu=0, so t=0 maps to periapsis (nu=0)
        pos = propagate_orbit(self.elem, times, mu=MU_SOLAR)
        # Verify positions match analytic orbit
        r = self.p / (1 + self.e * np.cos(nu_vals))
        expected_x = r * np.cos(nu_vals)
        expected_y = r * np.sin(nu_vals)
        np.testing.assert_allclose(pos[:, 0], expected_x, atol=1e-10)
        np.testing.assert_allclose(pos[:, 1], expected_y, atol=1e-10)
        np.testing.assert_allclose(pos[:, 2], 0.0, atol=1e-12)

    def test_velocity_periapsis_gt_apoapsis(self):
        v_peri = orbital_velocity(self.elem, 0.0, mu=MU_SOLAR)
        v_apo = orbital_velocity(self.elem, np.pi, mu=MU_SOLAR)
        assert np.linalg.norm(v_peri) > np.linalg.norm(v_apo)


# ── Degenerate case detection ────────────────────────────────────────

class TestDegenerateDetection:
    def test_collinear_points_2d(self):
        pts = np.array([[i, 2 * i] for i in range(5)], dtype=float)
        warnings = validate_points(pts)
        assert any("collinear" in w for w in warnings)

    def test_collinear_points_classify(self):
        pts = np.array([[i, 2 * i] for i in range(6)], dtype=float)
        coeffs = fit_conic(pts)
        assert classify_conic(coeffs) == "degenerate"

    def test_insufficient_points(self):
        pts = np.array([[0, 0], [1, 1], [2, 3]], dtype=float)
        warnings = validate_points(pts)
        assert any("5 points" in w for w in warnings)

    def test_duplicate_points(self):
        pts = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [7, 8]],
                        dtype=float)
        warnings = validate_points(pts)
        assert any("duplicate" in w for w in warnings)


# ── 3D orbit plane ───────────────────────────────────────────────────

class TestOrbitPlane3D:
    def test_plane_from_normal_roundtrip(self):
        # normal = [sin O sin i, -cos O sin i, cos i]
        inc_true = np.radians(45)
        Omega_true = np.radians(60)
        normal = np.array([
            np.sin(Omega_true) * np.sin(inc_true),
            -np.cos(Omega_true) * np.sin(inc_true),
            np.cos(inc_true),
        ])
        inc, Omega, e1, e2 = plane_from_normal(normal)
        assert abs(inc - inc_true) < 1e-12
        assert abs(Omega - Omega_true) < 1e-12
        # e1 and e2 should be orthonormal
        assert abs(np.dot(e1, e2)) < 1e-12
        assert abs(np.linalg.norm(e1) - 1) < 1e-12
        assert abs(np.linalg.norm(e2) - 1) < 1e-12

    def test_orbit_to_3d_flat(self):
        # inc=0, Omega=0 => orbit in xy-plane
        ox = np.array([1.0, 0.0, -1.0])
        oy = np.array([0.0, 1.0, 0.0])
        x3, y3, z3 = orbit_to_3d(ox, oy, 0.0, 0.0)
        np.testing.assert_allclose(x3, ox, atol=1e-12)
        np.testing.assert_allclose(y3, oy, atol=1e-12)
        np.testing.assert_allclose(z3, 0.0, atol=1e-12)

    def test_find_3d_foci_tilted_plane(self):
        # Generate an ellipse in a tilted plane
        inc = np.radians(45)
        Omega = np.radians(60)
        p, e, w = 2.0, 0.3, 0.5
        nu = np.linspace(0, 2 * np.pi, 9)[:-1]
        r = p / (1 + e * np.cos(nu))
        ox = r * np.cos(nu + w)
        oy = r * np.sin(nu + w)
        x3, y3, z3 = orbit_to_3d(ox, oy, inc, Omega)
        pts_3d = np.column_stack([x3, y3, z3])

        foci = find_3d_foci(pts_3d)
        assert len(foci) == 2
        # One focus should be near the origin (since orbit is around origin)
        dists = [np.linalg.norm(f) for f in foci]
        assert min(dists) < 0.5

    def test_orbit_to_3d_roundtrip(self):
        inc = np.radians(35)
        Omega = np.radians(110)
        _, e1, e2 = plane_from_normal(np.array([0, 0, 1]))[1:]
        inc2, Omega2, e1b, e2b = plane_from_normal(
            np.cross(
                np.array([np.cos(Omega), np.sin(Omega), 0]),
                np.array([-np.sin(Omega) * np.cos(inc),
                           np.cos(Omega) * np.cos(inc),
                           np.sin(inc)])
            )
        )
        # Project a point and lift back
        ox, oy = 3.0, 2.0
        x3, y3, z3 = orbit_to_3d(np.array([ox]), np.array([oy]),
                                   inc, Omega)
        _, _, e1, e2 = plane_from_normal(np.array([
            np.sin(Omega) * np.sin(inc),
            -np.cos(Omega) * np.sin(inc),
            np.cos(inc)]))
        pt3 = np.array([x3[0], y3[0], z3[0]])
        ox_back = np.dot(pt3, e1)
        oy_back = np.dot(pt3, e2)
        assert abs(ox_back - ox) < 1e-12
        assert abs(oy_back - oy) < 1e-12


# ── Kepler solvers ───────────────────────────────────────────────────

class TestKeplerSolvers:
    def test_nu_to_time_roundtrip_ellipse(self):
        e, p = 0.3, 2.0
        nu_orig = np.array([0.5, 1.0, 2.0, 4.0, 5.5])
        t = np.array([float(nu_to_time(v, e, p)) for v in nu_orig])
        ox, oy = time_to_pos(t, e, p, omega=0.0)
        nu_back = np.arctan2(oy, ox)
        # Wrap to [0, 2pi)
        nu_back = nu_back % (2 * np.pi)
        nu_check = nu_orig % (2 * np.pi)
        np.testing.assert_allclose(nu_back, nu_check, atol=1e-10)

    def test_orbit_type_classification(self):
        assert orbit_type(0.0) == "ellipse"
        assert orbit_type(0.5) == "ellipse"
        assert orbit_type(0.99) == "ellipse"
        assert orbit_type(1.0) == "parabola"
        assert orbit_type(1.5) == "hyperbola"
