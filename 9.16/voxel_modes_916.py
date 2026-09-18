"""Thermal voxel variance on the modes of a finite periodic comoving box.

``grid_size`` defines a top-hat voxel width, not a Fourier/Nyquist cutoff.
All lattice modes inside ``kmax`` are retained before instrument masking.
The input noise spectrum must precede this voxel averaging operation.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class _Geometry:
    k_perp: np.ndarray
    transverse_window_sum: np.ndarray
    transverse_mode_count: np.ndarray
    longitudinal_max_index: np.ndarray
    longitudinal_k: np.ndarray
    longitudinal_window: np.ndarray
    boxsize_mpc: float
    voxel_size_mpc: float
    fundamental_k: float


def _readonly(values):
    values.setflags(write=False)
    return values


@lru_cache(maxsize=8)
def _geometry(boxsize_mpc_h, grid_size, h_mock, kmax):
    """Group transverse modes with the same integer radius squared."""
    length = boxsize_mpc_h / h_mock
    fundamental = 2.0 * np.pi / length
    radius = kmax / fundamental
    # Account only for roundoff at an exactly specified spherical boundary.
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, radius**2)
    nmax = int(np.floor(np.sqrt(radius**2 + tolerance)))
    indices = np.arange(-nmax, nmax + 1, dtype=np.int64)
    radius_squared = indices[:, None] ** 2 + indices[None, :] ** 2
    selected = radius_squared <= radius**2 + tolerance
    integer_radius = radius_squared[selected]

    # np.sinc(t) = sin(pi*t)/(pi*t); k_i * Delta / 2 = pi*n_i/N.
    one_axis_window = np.sinc(indices / grid_size) ** 2
    transverse_window = (
        one_axis_window[:, None] * one_axis_window[None, :]
    )[selected]
    counts = np.bincount(integer_radius)
    weights = np.bincount(integer_radius, weights=transverse_window)
    present = np.flatnonzero(counts)
    max_parallel = np.floor(
        np.sqrt(np.maximum(0.0, radius**2 - present + tolerance))
    ).astype(np.int64)
    parallel_indices = np.arange(nmax + 1, dtype=np.int64)

    return _Geometry(
        k_perp=_readonly(fundamental * np.sqrt(present)),
        transverse_window_sum=_readonly(weights[present]),
        transverse_mode_count=_readonly(counts[present]),
        longitudinal_max_index=_readonly(max_parallel),
        longitudinal_k=_readonly(fundamental * parallel_indices),
        longitudinal_window=_readonly(np.sinc(parallel_indices / grid_size) ** 2),
        boxsize_mpc=length,
        voxel_size_mpc=length / grid_size,
        fundamental_k=fundamental,
    )


class FiniteBoxVoxelIntegrator:
    """Sum an axisymmetric, separable thermal noise spectrum efficiently.

    Parameters use comoving units: ``boxsize_mpc_h`` is in h^-1 Mpc and
    ``kmax`` is in Mpc^-1. No additional Nyquist cutoff is applied: modes
    that would alias on a grid_size^3 sampled map still contribute to the
    variance of the underlying continuous field's cube averages.

    The supported noise model is
        P_N(p, ell) = power_perp(p) * exp((ell / sigma_parallel)**2).
    This is the channel response used by baofisher's interferometer model.
    Both signs of all three Fourier coordinates are counted explicitly or
    through their exact degeneracy.  There is no extra real-field factor 1/2
    in a field variance (that factor instead appears in Fisher information).
    """

    def __init__(
        self,
        boxsize_mpc_h=3000.0,
        grid_size=512,
        h_mock=0.6766,
        kmax=0.535,
    ):
        if not all(np.isfinite(v) and v > 0 for v in (boxsize_mpc_h, h_mock, kmax)):
            raise ValueError("Box length, h_mock and kmax must be finite and positive.")
        if isinstance(grid_size, (bool, np.bool_)) or int(grid_size) != grid_size or grid_size <= 0:
            raise ValueError("grid_size must be a positive integer.")
        self.boxsize_mpc_h = float(boxsize_mpc_h)
        self.grid_size = int(grid_size)
        self.h_mock = float(h_mock)
        self.kmax = float(kmax)
        self._geometry = _geometry(
            self.boxsize_mpc_h, self.grid_size, self.h_mock, self.kmax
        )

    @property
    def k_perp(self):
        """Unique transverse lattice radii in Mpc^-1, including p=0."""
        return self._geometry.k_perp

    @property
    def fundamental_k(self):
        return self._geometry.fundamental_k

    @property
    def voxel_size_mpc(self):
        return self._geometry.voxel_size_mpc

    def integrate(self, power_perp, sigma_parallel, kpar_min, valid_perp=None):
        """Return voxel variance and coverage diagnostics.

        ``power_perp`` is P_N(p, ell=0), in mK^2 Mpc^3, evaluated at
        ``self.k_perp``.  It must exclude foreground and nonlinear cuts.
        ``valid_perp`` is an explicit instrument-support mask.  Invalid
        entries may contain infinity/NaN, since they are never multiplied
        into the integrand.  Valid entries must be finite and nonnegative.
        ``kpar_min`` applies to |k_parallel|.  Zero includes the ell=0 plane
        once; otherwise the positive and negative planes are both summed.

        This finite-box sum is not a continuum integral over a noise model
        that diverges at an interpolated zero-density edge.  It does not
        impose an arbitrary baseline-density floor or cap the noise.
        """
        if not np.isfinite(kpar_min) or kpar_min < 0:
            raise ValueError("kpar_min must be finite and nonnegative.")
        if np.isnan(sigma_parallel) or sigma_parallel <= 0:
            raise ValueError("sigma_parallel must be positive; infinity is allowed.")
        geom = self._geometry
        power = np.asarray(power_perp, dtype=float)
        if power.shape != self.k_perp.shape:
            raise ValueError("power_perp must have the same shape as k_perp.")
        valid = (
            np.ones(power.shape, dtype=bool)
            if valid_perp is None else np.asarray(valid_perp, dtype=bool)
        )
        if valid.shape != power.shape:
            raise ValueError("valid_perp must have the same shape as k_perp.")

        # A left-inclusive cutoff retains modes exactly at the boundary.
        cutoff = kpar_min / geom.fundamental_k
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, cutoff)
        minimum_index = max(0, int(np.ceil(cutoff - tolerance)))
        permitted = geom.longitudinal_max_index >= minimum_index
        selected = valid & permitted
        if np.any(~np.isfinite(power[selected])) or np.any(power[selected] < 0):
            raise ValueError("Noise power must be finite and nonnegative on supported modes.")

        # Count modes after the radial cutoff, before/after baseline masking.
        max_index = geom.longitudinal_max_index
        if minimum_index == 0:
            parallel_count = 2 * max_index + 1
        else:
            parallel_count = 2 * np.maximum(0, max_index - minimum_index + 1)
        mode_count_all = int(np.sum(geom.transverse_mode_count * parallel_count))
        mode_count = int(np.sum(
            geom.transverse_mode_count[selected] * parallel_count[selected]
        ))

        variance = 0.0
        if np.any(selected):
            highest = int(np.max(max_index[selected]))
            ell = geom.longitudinal_k[:highest + 1]
            with np.errstate(over="raise", invalid="raise"):
                try:
                    axial = geom.longitudinal_window[:highest + 1] * np.exp(
                        (ell / sigma_parallel) ** 2
                    )
                    axial *= 2.0
                    axial[0] *= 0.5
                    axial[:minimum_index] = 0.0
                    cumulative = np.cumsum(axial)
                    variance = float(np.sum(
                        power[selected]
                        * geom.transverse_window_sum[selected]
                        * cumulative[max_index[selected]]
                    ) / geom.boxsize_mpc**3)
                except FloatingPointError as error:
                    raise ValueError("Channel response or noise sum overflowed on valid modes.") from error
        if not np.isfinite(variance):
            raise ValueError("The voxel variance is not finite on the retained modes.")

        return {
            "variance_mK2": variance,
            "rms_mK": float(np.sqrt(variance)),
            "mode_count": mode_count,
            "mode_count_before_baseline_mask": mode_count_all,
            "supported_mode_fraction": (
                mode_count / mode_count_all if mode_count_all else 0.0
            ),
            "kpar_min_requested_Mpc_inv": float(kpar_min),
            "kpar_min_lattice_Mpc_inv": minimum_index * geom.fundamental_k,
            "kmax_Mpc_inv": self.kmax,
            "fundamental_k_Mpc_inv": geom.fundamental_k,
            "boxsize_Mpc": geom.boxsize_mpc,
            "voxel_size_Mpc": geom.voxel_size_mpc,
            "variance_definition": "finite_periodic_box_voxel_average",
            "nyquist_cut_applied": False,
        }
