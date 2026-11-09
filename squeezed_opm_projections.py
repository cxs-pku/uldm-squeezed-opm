#!/usr/bin/env python3
"""Simulations, figures, and data for the squeezed-light OPM ULDM study.

Usage: python codes/squeezed_opm_projections.py [--figures-only]
Run from the repository root. The --figures-only mode skips the CSV and
macro outputs and only regenerates the eight figure files.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator
from matplotlib.transforms import Bbox
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincinv
from scipy.stats import chi2, ncx2, norm


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "data"
BUILD = ROOT / "build"
SEED = 20260822

# Unboosted isotropic Maxwellian halo. The gradient-weighted axion-wind line is
# a gamma profile with shape 5/2 instead of the scalar-field shape 3/2.
V0_DM = 220_000.0 / 299_792_458.0
MAXWELLIAN_GRADIENT_SHAPE = 2.5
LINE_TAIL_PROBABILITY = 1.0e-12
T_PROJECTION = 1.0e5
SCAN_F_MIN = 1.0e-2
SCAN_F_MAX = 1.0e4
SCAN_OVERSAMPLING = 4

# Unknown-phase radial statistic. Its Rayleigh tail matches a one-sided
# Gaussian 5-sigma event.
LOCAL_Q5 = math.sqrt(-2.0 * math.log(0.5 * math.erfc(5.0 / math.sqrt(2.0))))
LOCAL_FALSE_ALARM = 0.5 * math.erfc(5.0 / math.sqrt(2.0))
DETECTION_PROBABILITY = 0.50

# Scan-wide Bonferroni bound over the positive-frequency template grid.
N_SCAN_TRIALS = int(
    math.ceil(SCAN_OVERSAMPLING * T_PROJECTION * (SCAN_F_MAX - SCAN_F_MIN))
)
GLOBAL_FALSE_ALARM = LOCAL_FALSE_ALARM / N_SCAN_TRIALS
GLOBAL_GAUSSIAN_Z = float(norm.isf(GLOBAL_FALSE_ALARM))

# Physical constants for the axion-electron benchmark.
RHO_DM_GEV_CM3 = 0.40
CM_IN_EV_INV = 5.067730716e4
M_E_EV = 510_998.95069
ELECTRON_G_FACTOR = 2.00231930436256
MU_B_EV_T = 5.7883817982e-5
PLANCK_EV_S = 4.135667696e-15

# Fiducial design point. The source level is the analytic optimum of the
# phase-jitter covariance model.
ETA_FID = 0.97
PHASE_FID = 0.008
BACKACTION_STRESS_MAX = 0.30
CORRELATION_STRESS_MAX = 0.50
CALIBRATION_MC_DRAWS = 500_000
NOISE_LEVELS = {
    # Output-referred polarization/readout noise amplitudes.
    "shot_out": 0.50,
    "laser_out": 0.10,
    # Input-referred magnetic-field noise amplitudes. The atomic term is the
    # Bloch-process noise after conversion by (gamma P0)^(-2).
    "atom_in": 0.20,
    "environment_in": 0.12,
    "light_shift_in": 0.05,
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
BLACK = "#222222"
GREEN = "#009E73"
MAGENTA = "#CC79A7"
GOLD = "#E69F00"
GRAY = "#888888"
PALE_BLUE = "#A6CEE3"
FIGURE_SIZE = (178.0 / 25.4, 120.0 / 25.4)
SHARED_GAIN_TICKS = (0.0, 0.5, 1.0, 1.5, 2.0)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.facecolor": "white",
            "legend.edgecolor": GRAY,
            "legend.fancybox": False,
        }
    )


def format_figure(fig: plt.Figure) -> None:
    """Pin labels, panel tags, and colorbars after the layout engine runs."""
    fig.set_facecolor("white")
    try:
        fig.set_constrained_layout(True)
        fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.14, wspace=0.10, hspace=0.15)
    except Exception:
        pass
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("black")
        ax.tick_params(which="major", direction="in", length=3.5, width=0.7)
        ax.tick_params(which="minor", direction="in", length=2.0, width=0.7)
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontsize(7)
        if ax.get_label() == "<colorbar>":
            if not isinstance(ax.yaxis.get_major_locator(), FixedLocator):
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=7)
            ax.yaxis.label.set_size(8)
        for legend in (
            artist
            for artist in ax.get_children()
            if isinstance(artist, matplotlib.legend.Legend)
        ):
            legend.set_frame_on(True)
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor(GRAY)
            legend.get_frame().set_linewidth(0.5)
            legend.get_frame().set_alpha(0.8)
            for text in legend.get_texts():
                text.set_fontsize(7)

    # The layout engine moves labels around. Draw once, read the rendered
    # positions, then pin them and disable the engine.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    columns: dict[int, list[plt.Axes]] = {}
    rows: dict[int, list[plt.Axes]] = {}
    for ax in axes:
        bbox = ax.get_window_extent(renderer)
        columns.setdefault(round(bbox.x0), []).append(ax)
        rows.setdefault(round(bbox.y0), []).append(ax)
    panel_left: dict[int, float] = {}

    # Left y-labels in one column share a left edge.
    for group in columns.values():
        labeled = [
            ax
            for ax in group
            if ax.yaxis.get_label_position() == "left" and ax.get_ylabel()
        ]
        if len(labeled) < 2:
            continue
        target_left = min(
            ax.yaxis.label.get_window_extent(renderer).x0 for ax in labeled
        )
        for ax in labeled:
            extent = ax.yaxis.label.get_window_extent(renderer)
            bbox = ax.get_window_extent(renderer)
            # set_label_coords anchors the right edge of a rotated label.
            right = target_left + extent.width
            ax.yaxis.set_label_coords((right - bbox.x0) / bbox.width, 0.5)
        for ax in group:
            panel_left[id(ax)] = target_left

    # X-labels in one row share a top edge.
    for group in rows.values():
        labeled = [ax for ax in group if ax.get_xlabel()]
        if len(labeled) < 2:
            continue
        target_top = min(
            ax.xaxis.label.get_window_extent(renderer).y0 for ax in labeled
        )
        for ax in labeled:
            bbox = ax.get_window_extent(renderer)
            ax.xaxis.set_label_coords(0.5, (target_top - bbox.y0) / bbox.height)

    # Panel tags go in the top-left corner, flush with the y-label edge.
    for ax in axes:
        panels = [text for text in ax.texts if text.get_gid() == "panel-label"]
        if not panels:
            continue
        bbox = ax.get_window_extent(renderer)
        label_extent = ax.yaxis.label.get_window_extent(renderer)
        left = panel_left.get(id(ax), label_extent.x0)
        x_fraction = (left - bbox.x0) / bbox.width
        lift = 4.0 + max(0.0, label_extent.y1 - bbox.y1)
        y_fraction = 1.0 + lift * fig.dpi / 72.0 / bbox.height
        for text in panels:
            text.set_transform(ax.transAxes)
            text.set_position((x_fraction, y_fraction))

    # Colorbar sits 2 pt right of its parent axes, vertically centered.
    for cax in fig.axes:
        if cax.get_label() != "<colorbar>":
            continue
        info = getattr(cax, "_colorbar_info", None)
        if info is not None and info.get("parents"):
            parents = info["parents"]
        else:
            colorbar = getattr(cax, "_colorbar", None) or getattr(cax, "colorbar", None)
            parents = [colorbar.mappable.axes] if colorbar is not None else []
        if not parents:
            continue
        parent = parents[-1]
        parent_extent = parent.get_window_extent(renderer)
        cbar_extent = cax.get_window_extent(renderer)
        shift_x = parent_extent.x1 + 2.0 * fig.dpi / 72.0 - cbar_extent.x0
        shift_y = (
            (parent_extent.y0 + parent_extent.y1) / 2.0
            - (cbar_extent.y0 + cbar_extent.y1) / 2.0
        )
        if abs(shift_x) > 0.5 or abs(shift_y) > 0.5:
            position = cax.get_position()
            position.x0 += shift_x / fig.bbox.width
            position.x1 += shift_x / fig.bbox.width
            position.y0 += shift_y / fig.bbox.height
            position.y1 += shift_y / fig.bbox.height
            cax.set_position(position)
            cax.set_axes_locator(None)
    fig.set_layout_engine(None)
    fig.canvas.draw()


def label_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
        gid="panel-label",
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    format_figure(fig)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    width = FIGURE_SIZE[0]
    height = max(FIGURE_SIZE[1], tight.height + 0.06)
    export_box = Bbox.from_bounds(
        tight.x0 + 0.5 * (tight.width - width),
        tight.y0 + 0.5 * (tight.height - height),
        width,
        height,
    )
    fig.set_layout_engine(None)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches=export_box)
    fig.savefig(FIG / f"{name}.svg", bbox_inches=export_box)
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches=export_box)
    plt.close(fig)


def effective_variance(r: float, eta: float, phase_rms: float) -> float:
    """Detected squeezed-quadrature variance after loss and phase jitter."""
    # Exact covariance average for a zero-mean Gaussian analyzer angle.
    c2 = 0.5 * (1.0 + math.exp(-2.0 * phase_rms**2))
    v_phase = math.exp(-2.0 * r) * c2 + math.exp(2.0 * r) * (1.0 - c2)
    return eta * v_phase + (1.0 - eta)


def normalized_backaction_psd(
    r: float,
    epsilon_ba: float,
    optical_fraction: float,
    *,
    r_reference: float | None = None,
    eta: float = ETA_FID,
    phase_rms: float = PHASE_FID,
) -> float:
    """Excess backaction PSD normalized to the coherent baseline total PSD."""
    if not np.isfinite(r) or r < 0.0:
        raise ValueError("r must be finite and nonnegative")
    if not np.isfinite(epsilon_ba) or epsilon_ba < 0.0:
        raise ValueError("epsilon_ba must be finite and nonnegative")
    if not np.isfinite(optical_fraction) or not 0.0 <= optical_fraction <= 1.0:
        raise ValueError("optical_fraction must lie between zero and one")
    if r_reference is None:
        r_reference = optimum_squeezing_r(PHASE_FID)
    reference_psd = (
        optical_fraction * effective_variance(r_reference, eta, phase_rms)
        + 1.0
        - optical_fraction
    )
    denominator = math.exp(2.0 * r_reference) - 1.0
    return epsilon_ba * reference_psd * (math.exp(2.0 * r) - 1.0) / denominator


def stressed_gain(
    r: float,
    epsilon_ba: float,
    optical_fraction: float,
    *,
    eta: float = ETA_FID,
    phase_rms: float = PHASE_FID,
) -> float:
    """Amplitude gain with the phenomenological excess-backaction stress."""
    denominator = (
        optical_fraction * effective_variance(r, eta, phase_rms)
        + 1.0
        - optical_fraction
        + normalized_backaction_psd(
            r,
            epsilon_ba,
            optical_fraction,
            eta=eta,
            phase_rms=phase_rms,
        )
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("Backaction-stressed PSD is not positive")
    return 1.0 / math.sqrt(denominator)


def optimum_squeezing_r(phase_rms: float) -> float:
    """Analytic minimizer of the detected variance at fixed phase jitter."""
    if not np.isfinite(phase_rms) or phase_rms < 0.0:
        raise ValueError("phase_rms must be a finite, nonnegative scalar")
    c2 = 0.5 * (1.0 + math.exp(-2.0 * phase_rms**2))
    if c2 >= 1.0 - 1.0e-15:
        # Zero phase jitter: the variance decreases monotonically in r.
        return math.inf
    return 0.25 * math.log(c2 / (1.0 - c2))


R_FID = optimum_squeezing_r(PHASE_FID)


def stressed_optimum_r(epsilon_ba: float, optical_fraction: float) -> float:
    """Analytic total-noise optimum under the normalized backaction stress."""
    if not np.isfinite(epsilon_ba) or epsilon_ba < 0.0:
        raise ValueError("epsilon_ba must be finite and nonnegative")
    if not np.isfinite(optical_fraction) or not 0.0 < optical_fraction <= 1.0:
        raise ValueError("optical_fraction must lie in (0, 1]")
    c2 = 0.5 * (1.0 + math.exp(-2.0 * PHASE_FID**2))
    reference_psd = (
        optical_fraction * effective_variance(R_FID, ETA_FID, PHASE_FID)
        + 1.0
        - optical_fraction
    )
    beta = epsilon_ba * reference_psd / (math.exp(2.0 * R_FID) - 1.0)
    numerator = optical_fraction * ETA_FID * c2
    denominator = optical_fraction * ETA_FID * (1.0 - c2) + beta
    if numerator <= denominator:
        return 0.0
    return 0.25 * math.log(numerator / denominator)


def correlated_total_psd(
    frequency: float,
    r: float,
    correlation_magnitude: float,
    correlation_phase: float,
    *,
    epsilon_ba: float = 0.0,
) -> float:
    """Shot/light-shift PSD with a bounded complex correlation.

    The output-shot/input-light-shift covariance is positive semidefinite for
    ``0 <= correlation_magnitude <= 1``. The complex OPM phase is kept in the
    input-equivalent cross term.
    """
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be finite and positive")
    if not np.isfinite(correlation_magnitude) or not 0.0 <= correlation_magnitude <= 1.0:
        raise ValueError("correlation_magnitude must lie between zero and one")
    base = model_noise(np.asarray([frequency], dtype=float), 1.0)
    shot_psd = float(base["shot"][0] ** 2)
    light_shift_psd = float(base["light_shift"][0] ** 2)
    coherent_total_psd = float(base["total_coherent"][0] ** 2)
    optical_fraction = shot_psd / coherent_total_psd
    variance = effective_variance(r, ETA_FID, PHASE_FID)
    nonshot_psd = coherent_total_psd - shot_psd
    excess_ba = (
        normalized_backaction_psd(r, epsilon_ba, optical_fraction)
        * coherent_total_psd
    )
    response_phase = float(np.angle(normalized_transfer(np.asarray([frequency]))[0]))
    cross = (
        2.0
        * correlation_magnitude
        * math.sqrt(variance * shot_psd * light_shift_psd)
        * math.cos(correlation_phase - response_phase)
    )
    total = variance * shot_psd + nonshot_psd + excess_ba + cross
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Correlated total PSD is not positive")
    return total


def normalized_transfer(freq: np.ndarray) -> np.ndarray:
    """Complex single-pole response for the e^{-2 pi i f t} convention."""
    f = np.asarray(freq, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("Frequency values must be finite and nonnegative")
    return 1.0 / (1.0 + 1j * f / 135.0)


def coherence_time(freq: np.ndarray) -> np.ndarray:
    """Inverse-linewidth scale tau_DM = 1/(f*v0^2) = 1/(2*Delta_f)."""
    f = np.asarray(freq, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0.0):
        raise ValueError("ULDM coherence frequencies must be finite and positive")
    return 1.0 / (f * V0_DM**2)


def stochastic_line_factor(freq: np.ndarray, observation_time: float) -> np.ndarray:
    """ULDM Asimov line factor relative to a deterministic rms-tone target."""
    if not np.isfinite(observation_time) or observation_time <= 0.0:
        raise ValueError("observation_time must be finite and positive")
    return stochastic_line_coefficient(freq, observation_time) / (
        global_gaussian_target(observation_time) / math.sqrt(2.0)
    )


def scan_trial_count(observation_time: float) -> int:
    """Bonferroni count for the oversampled positive-frequency template grid."""
    if not np.isfinite(observation_time) or observation_time <= 0.0:
        raise ValueError("observation_time must be finite and positive")
    return max(
        1,
        int(
            math.ceil(
                SCAN_OVERSAMPLING
                * observation_time
                * (SCAN_F_MAX - SCAN_F_MIN)
            )
        ),
    )


def global_false_alarm_probability(observation_time: float) -> float:
    """Per-template false-alarm probability for the global 5-sigma target."""
    return LOCAL_FALSE_ALARM / scan_trial_count(observation_time)


def global_gaussian_target(observation_time: float) -> float:
    """Local Gaussian Z matching the scan-wide Bonferroni target."""
    return float(norm.isf(global_false_alarm_probability(observation_time)))


def maxwellian_gradient_bin_weights(
    frequency: float, observation_time: float
) -> np.ndarray:
    """Gamma (5/2) line integrated over FFT bins of width 1/T.

    The tail beyond an explicit 1e-12 truncation is folded into the last bin
    so the weights sum to unity.
    """
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be finite and positive")
    if not np.isfinite(observation_time) or observation_time <= 0.0:
        raise ValueError("observation_time must be finite and positive")
    scale = frequency * V0_DM**2 / 2.0
    bin_width = 1.0 / observation_time
    upper = (
        float(gammaincinv(MAXWELLIAN_GRADIENT_SHAPE, 1.0 - LINE_TAIL_PROBABILITY))
        * scale
    )
    n_bins = max(1, int(math.ceil(upper / bin_width)))
    edges = np.arange(n_bins + 1, dtype=float) * bin_width
    cdf = gammainc(MAXWELLIAN_GRADIENT_SHAPE, edges / scale)
    weights = np.diff(cdf)
    weights[-1] += 1.0 - float(cdf[-1])
    if (
        weights.ndim != 1
        or weights.size != n_bins
        or np.any(~np.isfinite(weights))
        or np.any(weights < -1.0e-14)
        or abs(float(np.sum(weights)) - 1.0) > 1.0e-12
    ):
        raise RuntimeError("Invalid Maxwellian line-bin weights")
    return np.maximum(weights, 0.0) / float(np.sum(np.maximum(weights, 0.0)))


def line_bin_count(freq: np.ndarray, observation_time: float) -> np.ndarray:
    """Number of retained FFT bins in each integrated ULDM line template."""
    f = np.asarray(freq, dtype=float)
    return np.asarray(
        [maxwellian_gradient_bin_weights(float(ff), observation_time).size for ff in f],
        dtype=int,
    )


def whittle_asimov_q(power_ratio: float, weights: np.ndarray) -> float:
    """Expected log-likelihood statistic for a stochastic excess-power template."""
    if not np.isfinite(power_ratio) or power_ratio < 0.0:
        raise ValueError("power_ratio must be finite and nonnegative")
    x = power_ratio * np.asarray(weights, dtype=float)
    return float(2.0 * np.sum(x - np.log1p(x)))


def finite_window_eigenfractions(
    time_bandwidth: float, n_quad_override: int | None = None
) -> np.ndarray:
    """Eigenvalue fractions of the finite-window Maxwellian covariance.

    The normalized covariance is R(s-s') = [1 - i*2*pi*u*(s-s')]^(-5/2) on the
    unit interval with u = T*Delta_f. Gauss-Legendre Nystrom quadrature gives a
    positive semidefinite approximation with unit trace.
    """
    if not np.isfinite(time_bandwidth) or time_bandwidth < 0.0:
        raise ValueError("time_bandwidth must be finite and nonnegative")
    n_quad = (
        max(48, int(math.ceil(20.0 * time_bandwidth)))
        if n_quad_override is None
        else int(n_quad_override)
    )
    if n_quad < 2:
        raise ValueError("n_quad_override must be at least two")
    nodes, weights = leggauss(n_quad)
    sample = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    lag = sample[:, None] - sample[None, :]
    covariance = (1.0 - 1j * 2.0 * np.pi * time_bandwidth * lag) ** (
        -MAXWELLIAN_GRADIENT_SHAPE
    )
    matrix = np.sqrt(weights[:, None] * weights[None, :]) * covariance
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.min(eigenvalues) < -100.0 * tolerance:
        raise RuntimeError("Finite-window ULDM covariance is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalues = eigenvalues[eigenvalues > tolerance]
    total = float(np.sum(eigenvalues))
    if not np.isfinite(total) or abs(total - 1.0) > 5.0e-10:
        raise RuntimeError("Finite-window ULDM covariance has invalid trace")
    return eigenvalues / total


def validate_nystrom_convergence() -> tuple[bool, float]:
    """Compare finite-window thresholds against a doubled quadrature order."""
    target = global_gaussian_target(T_PROJECTION) ** 2
    relative_shifts = []
    for time_bandwidth in (0.1, 1.0, 8.0, 11.0, 13.0):
        n_quad = max(48, int(math.ceil(20.0 * time_bandwidth)))
        coarse = finite_window_eigenfractions(time_bandwidth, n_quad)
        fine = finite_window_eigenfractions(time_bandwidth, 2 * n_quad)
        coarse_power = required_power_ratio_from_modes(coarse, target)
        fine_power = required_power_ratio_from_modes(fine, target)
        relative_shifts.append(abs(math.sqrt(coarse_power / fine_power) - 1.0))
        if abs(float(np.sum(coarse)) - 1.0) > 5.0e-10 or abs(float(np.sum(fine)) - 1.0) > 5.0e-10:
            return False, float(max(relative_shifts))
    maximum = float(max(relative_shifts))
    return bool(np.isfinite(maximum) and maximum < 1.0e-3), maximum


def required_power_ratio_from_modes(modes: np.ndarray, target_q: float) -> float:
    """Solve the covariance Asimov equation for normalized eigenmodes."""
    eigenfractions = np.asarray(modes, dtype=float)

    def objective(power_ratio: float) -> float:
        return whittle_asimov_q(power_ratio, eigenfractions) - target_q

    upper = 1.0
    while objective(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0e12:
            raise RuntimeError("Failed to bracket the stochastic ULDM threshold")
    value = float(brentq(objective, 0.0, upper, xtol=1.0e-11, rtol=1.0e-12))
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError("Invalid stochastic ULDM threshold")
    return value


@functools.lru_cache(maxsize=None)
def required_stochastic_power_ratio(
    frequency: float, observation_time: float
) -> float:
    """Solve q_A = Z_global^2 for B_rms^2*T/S at a ULDM frequency."""
    target_q = global_gaussian_target(observation_time) ** 2
    time_bandwidth = observation_time * frequency * V0_DM**2 / 2.0

    # Finite-window covariance below u=13, Whittle bins above u=11, cubic
    # smoothstep in between.
    if time_bandwidth < 13.0:
        finite_modes = finite_window_eigenfractions(time_bandwidth)
        finite_power = required_power_ratio_from_modes(finite_modes, target_q)
    else:
        finite_power = math.nan
    if time_bandwidth > 11.0:
        whittle_modes = maxwellian_gradient_bin_weights(frequency, observation_time)
        whittle_power = required_power_ratio_from_modes(whittle_modes, target_q)
    else:
        whittle_power = math.nan

    if time_bandwidth <= 11.0:
        return finite_power
    if time_bandwidth >= 13.0:
        return whittle_power
    fraction = (time_bandwidth - 11.0) / 2.0
    smooth = fraction**2 * (3.0 - 2.0 * fraction)
    return (1.0 - smooth) * finite_power + smooth * whittle_power


def stochastic_line_coefficient(
    freq: np.ndarray, observation_time: float
) -> np.ndarray:
    """Coefficient multiplying sqrt(S1/T) for the Maxwellian ULDM line."""
    f = np.asarray(freq, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0.0):
        raise ValueError("ULDM frequencies must be finite and positive")
    return np.asarray(
        [
            math.sqrt(required_stochastic_power_ratio(float(ff), observation_time))
            for ff in f
        ],
        dtype=float,
    )


@functools.lru_cache(maxsize=None)
def required_noncentrality(false_alarm: float = LOCAL_FALSE_ALARM) -> float:
    """Noncentrality for 50% detection of the two-quadrature power statistic."""
    if not np.isfinite(false_alarm) or not 0.0 < false_alarm < 1.0:
        raise ValueError("false_alarm must lie strictly between zero and one")
    dof = 2
    critical = float(chi2.isf(false_alarm, dof))

    def objective(noncentrality: float) -> float:
        return float(ncx2.sf(critical, dof, noncentrality) - DETECTION_PROBABILITY)

    upper = max(critical, 1.0)
    while objective(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0e9:
            raise RuntimeError("Failed to bracket the noncentral chi-square threshold")
    value = float(brentq(objective, 0.0, upper, xtol=1.0e-11, rtol=1.0e-12))
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError("Invalid noncentrality returned by threshold calibration")
    return value


def ul_dm_asimov_target(
    noise_amplitude: np.ndarray, freq: np.ndarray, observation_time: float
) -> np.ndarray:
    """Scan-wide Gaussian-equivalent Maxwellian-ULDM expected-LLR target."""
    noise_amplitude = np.asarray(noise_amplitude, dtype=float)
    freq = np.asarray(freq, dtype=float)
    if noise_amplitude.shape != freq.shape:
        raise ValueError("Noise and frequency arrays must have identical shapes")
    return (
        stochastic_line_coefficient(freq, observation_time)
        * noise_amplitude
        / np.sqrt(observation_time)
    )


def deterministic_asimov_target(
    noise_amplitude: np.ndarray, observation_time: float
) -> np.ndarray:
    """Deterministic rms field at the same Gaussian-equivalent Asimov target."""
    noise_amplitude = np.asarray(noise_amplitude, dtype=float)
    if np.any(~np.isfinite(noise_amplitude)) or np.any(noise_amplitude < 0.0):
        raise ValueError("Noise amplitudes must be finite and nonnegative")
    if not np.isfinite(observation_time) or observation_time <= 0.0:
        raise ValueError("observation_time must be finite and positive")
    coefficient = global_gaussian_target(observation_time) / math.sqrt(2.0)
    return coefficient * noise_amplitude / np.sqrt(observation_time)


def unresolved_exact_to_asimov_amplitude_factor(observation_time: float) -> float:
    """Exact-power/Asimov amplitude ratio for one circular-complex ULDM mode."""
    false_alarm = global_false_alarm_probability(observation_time)
    critical_power = -math.log(false_alarm)
    exact_power_ratio = critical_power / (-math.log(DETECTION_PROBABILITY)) - 1.0
    asimov_power_ratio = required_power_ratio_from_modes(
        np.asarray([1.0]), global_gaussian_target(observation_time) ** 2
    )
    return math.sqrt(exact_power_ratio / asimov_power_ratio)


def modes_from_time_bandwidth(
    time_bandwidth: float, observation_time: float = T_PROJECTION
) -> np.ndarray:
    """Normalized ULDM covariance modes for a selected T*Delta_f."""
    if not np.isfinite(time_bandwidth) or time_bandwidth < 0.0:
        raise ValueError("time_bandwidth must be finite and nonnegative")
    if time_bandwidth <= 11.0:
        return finite_window_eigenfractions(time_bandwidth)
    frequency = 2.0 * time_bandwidth / (observation_time * V0_DM**2)
    if time_bandwidth >= 13.0:
        return maxwellian_gradient_bin_weights(frequency, observation_time)
    finite = finite_window_eigenfractions(time_bandwidth)
    whittle = maxwellian_gradient_bin_weights(frequency, observation_time)
    # Modes of different dimension cannot be interpolated one by one. Use the
    # closer representation on either side of the checked midpoint.
    return finite if time_bandwidth < 12.0 else whittle


def _exp_sum_cgf(t: float, weights: np.ndarray) -> tuple[float, float, float]:
    """CGF and first two derivatives of a weighted unit-exponential sum."""
    ww = np.asarray(weights, dtype=float)
    denominator = 1.0 - t * ww
    if np.any(denominator <= 0.0):
        raise ValueError("Saddlepoint lies outside the exponential CGF domain")
    scaled = ww / denominator
    return (
        float(-np.sum(np.log(denominator))),
        float(np.sum(scaled)),
        float(np.sum(scaled**2)),
    )


def _saddlepoint_parameter(value: float, weights: np.ndarray) -> float:
    """Solve K'(t) = value for a positive weighted exponential sum."""
    ww = np.asarray(weights, dtype=float)
    ww = ww[np.isfinite(ww) & (ww > 1.0e-15 * np.max(ww))]
    if ww.size == 0 or not np.isfinite(value) or value <= 0.0:
        raise ValueError("Invalid weighted-exponential saddlepoint input")
    mean = float(np.sum(ww))
    if abs(value / mean - 1.0) < 1.0e-12:
        return 0.0

    def objective(t: float) -> float:
        return _exp_sum_cgf(t, ww)[1] - value

    if value > mean:
        upper = (1.0 - 1.0e-12) / float(np.max(ww))
        return float(brentq(objective, 0.0, upper, xtol=1.0e-13, rtol=1.0e-13))
    lower = -1.0 / float(np.max(ww))
    while objective(lower) > 0.0:
        lower *= 2.0
        if lower < -1.0e16:
            raise RuntimeError("Failed to bracket the lower exponential saddlepoint")
    return float(brentq(objective, lower, 0.0, xtol=1.0e-13, rtol=1.0e-13))


def saddlepoint_survival(value: float, weights: np.ndarray) -> float:
    """Lugannani-Rice survival probability for a weighted exponential sum."""
    ww = np.asarray(weights, dtype=float)
    if ww.ndim != 1 or np.any(~np.isfinite(ww)) or np.any(ww < 0.0):
        raise ValueError("Exponential weights must be finite and nonnegative")
    if value <= 0.0:
        return 1.0
    tolerance = 1.0e-14 * max(1.0, float(np.max(ww)))
    ww = ww[ww > tolerance]
    if ww.size == 0:
        return 0.0
    if ww.size == 1:
        return float(math.exp(-value / float(ww[0])))
    t = _saddlepoint_parameter(value, ww)
    if abs(t) < 1.0e-9 / float(np.max(ww)):
        variance = float(np.sum(ww**2))
        return float(norm.sf((value - float(np.sum(ww))) / math.sqrt(variance)))
    cgf, _, second = _exp_sum_cgf(t, ww)
    signed_root = math.copysign(math.sqrt(max(0.0, 2.0 * (t * value - cgf))), t)
    u = t * math.sqrt(second)
    if abs(signed_root) < 1.0e-8 or abs(u) < 1.0e-8:
        variance = float(np.sum(ww**2))
        return float(norm.sf((value - float(np.sum(ww))) / math.sqrt(variance)))
    survival = float(
        norm.sf(signed_root)
        + norm.pdf(signed_root) * (1.0 / u - 1.0 / signed_root)
    )
    return min(1.0, max(np.finfo(float).tiny, survival))


def saddlepoint_quantile(weights: np.ndarray, survival_probability: float) -> float:
    """Upper-tail quantile of a weighted exponential sum."""
    if not np.isfinite(survival_probability) or not 0.0 < survival_probability < 1.0:
        raise ValueError("survival_probability must lie strictly between zero and one")
    ww = np.asarray(weights, dtype=float)
    mean = float(np.sum(ww))

    def objective(value: float) -> float:
        return math.log(saddlepoint_survival(value, ww)) - math.log(
            survival_probability
        )

    lower = max(np.finfo(float).tiny, 0.05 * mean)
    upper = max(mean, float(np.max(ww)))
    while objective(upper) > 0.0:
        upper *= 2.0
        if upper > 1.0e16 * max(mean, 1.0):
            raise RuntimeError("Failed to bracket the saddlepoint quantile")
    return float(brentq(objective, lower, upper, xtol=1.0e-10, rtol=1.0e-10))


def finite_sample_power_ratio_saddlepoint(
    modes: np.ndarray, survival_probability: float
) -> float:
    """50%-power signal ratio for the simple covariance likelihood."""
    fractions = np.asarray(modes, dtype=float)
    fractions = fractions[fractions > 1.0e-14 * np.max(fractions)]
    fractions = fractions / float(np.sum(fractions))
    if fractions.size == 1:
        return -math.log(survival_probability) / (-math.log(DETECTION_PROBABILITY)) - 1.0

    def detection_minus_half(power_ratio: float) -> float:
        excess = power_ratio * fractions
        null_weights = excess / (1.0 + excess)
        critical = saddlepoint_quantile(null_weights, survival_probability)
        return saddlepoint_survival(critical, excess) - DETECTION_PROBABILITY

    lower = 1.0e-8
    upper = 1.0
    while detection_minus_half(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0e10:
            raise RuntimeError("Failed to bracket the finite-sample ULDM calibration")
    return float(brentq(detection_minus_half, lower, upper, xtol=1.0e-8, rtol=1.0e-8))


def finite_sample_ratio_for_bandwidth(
    time_bandwidth: float,
    survival_probability: float,
    observation_time: float = T_PROJECTION,
) -> float:
    """Finite-mode/Whittle field correction for a line resolution.

    The unresolved end anchors to the analytic one-mode result, and a short
    smooth bridge joins the finite-window and resolved-bin approximations.
    """
    if (
        not np.isfinite(time_bandwidth)
        or time_bandwidth < 0.0
        or not np.isfinite(observation_time)
        or observation_time <= 0.0
    ):
        raise ValueError("Invalid line-resolution input")
    one_mode = unresolved_exact_to_asimov_amplitude_factor(observation_time)
    low_anchor, low_bridge = 0.01, 0.10
    if time_bandwidth <= low_anchor:
        return one_mode
    finite_modes = finite_window_eigenfractions(time_bandwidth)
    finite_asimov = required_power_ratio_from_modes(
        finite_modes, global_gaussian_target(observation_time) ** 2
    )
    finite_power = finite_sample_power_ratio_saddlepoint(
        finite_modes, survival_probability
    )
    finite_ratio = math.sqrt(finite_power / finite_asimov)
    if time_bandwidth < low_bridge:
        fraction = (time_bandwidth - low_anchor) / (low_bridge - low_anchor)
        smooth = fraction**2 * (3.0 - 2.0 * fraction)
        return (1.0 - smooth) * one_mode + smooth * finite_ratio
    if time_bandwidth <= 11.0:
        return finite_ratio

    frequency = 2.0 * time_bandwidth / (observation_time * V0_DM**2)
    whittle_modes = maxwellian_gradient_bin_weights(frequency, observation_time)
    whittle_asimov = required_power_ratio_from_modes(
        whittle_modes, global_gaussian_target(observation_time) ** 2
    )
    whittle_power = finite_sample_power_ratio_saddlepoint(
        whittle_modes, survival_probability
    )
    whittle_ratio = math.sqrt(whittle_power / whittle_asimov)
    if time_bandwidth >= 13.0:
        return whittle_ratio
    fraction = (time_bandwidth - 11.0) / 2.0
    smooth = fraction**2 * (3.0 - 2.0 * fraction)
    return (1.0 - smooth) * finite_ratio + smooth * whittle_ratio


def importance_sampled_tail(
    weights: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    draws: int = CALIBRATION_MC_DRAWS,
) -> tuple[float, float]:
    """Exponentially tilted Monte Carlo estimate of an extreme upper tail."""
    ww = np.asarray(weights, dtype=float)
    ww = ww[ww > 1.0e-14 * np.max(ww)]
    t = _saddlepoint_parameter(threshold, ww)
    if t <= 0.0:
        raise ValueError("Importance sampling requires an upper-tail threshold")
    cgf = _exp_sum_cgf(t, ww)[0]
    rates = 1.0 - t * ww
    total = 0.0
    total_sq = 0.0
    completed = 0
    batch_size = max(250, min(4000, 2_000_000 // ww.size))
    while completed < draws:
        batch = min(batch_size, draws - completed)
        exponentials = rng.exponential(scale=1.0 / rates, size=(batch, ww.size))
        values = exponentials @ ww
        contributions = np.where(
            values >= threshold,
            np.exp(cgf - t * values),
            0.0,
        )
        total += float(np.sum(contributions))
        total_sq += float(np.sum(contributions**2))
        completed += batch
    estimate = total / draws
    variance = max(0.0, total_sq / draws - estimate**2) / draws
    return estimate, math.sqrt(variance)


def _weighted_exponential_median(
    weights: np.ndarray, rng: np.random.Generator, draws: int = 300_000
) -> float:
    """Median of a weighted unit-exponential sum from fixed-seed batches."""
    ww = np.asarray(weights, dtype=float)
    values = np.empty(draws, dtype=float)
    batch_size = max(250, min(4000, 2_000_000 // ww.size))
    completed = 0
    while completed < draws:
        batch = min(batch_size, draws - completed)
        values[completed : completed + batch] = (
            rng.exponential(size=(batch, ww.size)) @ ww
        )
        completed += batch
    return float(np.median(values))


def finite_sample_np_ratio_importance(
    modes: np.ndarray,
    survival_probability: float,
    *,
    seed: int,
    draws: int = 120_000,
) -> tuple[float, float, float]:
    """Fixed-alternative median-power calibration with tilted sampling."""
    ww = np.asarray(modes, dtype=float)
    ww = ww[ww > 1.0e-14 * np.max(ww)]
    ww = ww / float(np.sum(ww))
    if ww.size == 1:
        exact_power = (
            -math.log(survival_probability) / (-math.log(DETECTION_PROBABILITY)) - 1.0
        )
        asimov_power = required_power_ratio_from_modes(
            ww, global_gaussian_target(T_PROJECTION) ** 2
        )
        return math.sqrt(exact_power / asimov_power), 1.0, 0.0

    median = _weighted_exponential_median(
        ww, np.random.default_rng(seed), draws=max(100_000, draws)
    )
    # Reuse the same unit-exponential variates at every root evaluation so the
    # root is deterministic.
    base_rng = np.random.default_rng(seed + 1_000_000)
    base = np.empty((draws, ww.size), dtype=float)
    batch_size = max(250, min(4000, 2_000_000 // ww.size))
    completed = 0
    while completed < draws:
        batch = min(batch_size, draws - completed)
        base[completed : completed + batch] = base_rng.exponential(
            size=(batch, ww.size)
        )
        completed += batch

    def tail_for_mu(mu: float) -> tuple[float, float]:
        coefficients = mu * ww / (1.0 + mu * ww)
        threshold = mu * median
        tilt = _saddlepoint_parameter(threshold, coefficients)
        rates = 1.0 - tilt * coefficients
        values = (base / rates) @ coefficients
        log_weight_normalization = _exp_sum_cgf(tilt, coefficients)[0]
        log_weights = log_weight_normalization - tilt * values
        selected = values >= threshold
        if not np.any(selected):
            return 0.0, 0.0
        contribution = np.where(selected, np.exp(log_weights), 0.0)
        estimate = float(np.mean(contribution))
        standard_error = float(np.std(contribution, ddof=1) / math.sqrt(draws))
        return estimate, standard_error

    def objective(mu: float) -> float:
        estimate, _ = tail_for_mu(mu)
        return math.log(max(estimate, np.finfo(float).tiny)) - math.log(
            survival_probability
        )

    lower = 1.0e-5
    upper = 1.0
    while objective(upper) > 0.0:
        upper *= 2.0
        if upper > 1.0e8:
            raise RuntimeError("Failed to bracket the fixed-alternative NP root")
    root = float(brentq(objective, lower, upper, xtol=2.0e-4, rtol=2.0e-6))
    tail_estimate, tail_error = tail_for_mu(root)
    asimov_power = required_power_ratio_from_modes(
        ww, global_gaussian_target(T_PROJECTION) ** 2
    )
    ratio = math.sqrt(root / asimov_power)
    return ratio, tail_estimate / survival_probability, tail_error / survival_probability


@functools.lru_cache(maxsize=1)
def finite_sample_calibration_curve() -> dict[str, np.ndarray | float]:
    """Asimov-to-finite-sample correction with tail-importance validation."""
    time_bandwidth = np.logspace(-4.0, 2.0, 52)
    alpha = global_false_alarm_probability(T_PROJECTION)
    saddle_ratio = []
    for value in time_bandwidth:
        saddle_ratio.append(
            finite_sample_ratio_for_bandwidth(float(value), alpha, T_PROJECTION)
        )

    validation_bandwidth = np.asarray(
        [
            1.0e-3,
            0.10,
            T_PROJECTION * 10.0 * V0_DM**2 / 2.0,
            3.0,
            T_PROJECTION * 300.0 * V0_DM**2 / 2.0,
        ]
    )
    validation_ratio = []
    validation_error = []
    validation_tail_ratio = []
    validation_tail_error = []
    validation_detection_ratio = []
    validation_detection_error = []
    for index, value in enumerate(validation_bandwidth):
        modes = modes_from_time_bandwidth(float(value))
        asimov_power = required_power_ratio_from_modes(
            modes, global_gaussian_target(T_PROJECTION) ** 2
        )
        validation_value, relative_tail, relative_error = (
            finite_sample_np_ratio_importance(
                modes,
                alpha,
                seed=SEED + 200 + index,
                draws=CALIBRATION_MC_DRAWS,
            )
        )
        validation_ratio.append(validation_value)
        finite_power = validation_value**2 * asimov_power
        amplitude_error = 0.5 * relative_error * validation_value
        validation_error.append(amplitude_error)
        exact_median = _weighted_exponential_median(
            modes,
            np.random.default_rng(SEED + 200 + index),
            draws=CALIBRATION_MC_DRAWS,
        )
        exact_coefficients = finite_power * modes / (1.0 + finite_power * modes)
        exact_threshold = finite_power * exact_median
        exact_tail_estimate, exact_tail_error = importance_sampled_tail(
            exact_coefficients,
            exact_threshold,
            np.random.default_rng(SEED + 500 + index),
            draws=CALIBRATION_MC_DRAWS,
        )
        validation_tail_ratio.append(exact_tail_estimate / alpha)
        validation_tail_error.append(exact_tail_error / alpha)
        # Simulate the alternative at the calibrated root to verify the
        # median-power construction.
        detection_rng = np.random.default_rng(SEED + 300 + index)
        excess = finite_power * modes
        detected = 0
        completed = 0
        batch_size = max(250, min(4000, 2_000_000 // excess.size))
        critical = finite_power * float(
            _weighted_exponential_median(
                modes,
                np.random.default_rng(SEED + 400 + index),
                draws=CALIBRATION_MC_DRAWS,
            )
        )
        while completed < CALIBRATION_MC_DRAWS:
            batch = min(batch_size, CALIBRATION_MC_DRAWS - completed)
            values = detection_rng.exponential(
                scale=1.0, size=(batch, excess.size)
            ) @ excess
            detected += int(np.count_nonzero(values >= critical))
            completed += batch
        detection_probability = detected / CALIBRATION_MC_DRAWS
        detection_standard_error = math.sqrt(
            detection_probability
            * (1.0 - detection_probability)
            / CALIBRATION_MC_DRAWS
        )
        validation_detection_ratio.append(
            detection_probability / DETECTION_PROBABILITY
        )
        validation_detection_error.append(
            detection_standard_error / DETECTION_PROBABILITY
        )
    combined_bandwidth = np.concatenate(
        [time_bandwidth, np.asarray(validation_bandwidth, dtype=float)]
    )
    combined_ratio = np.concatenate(
        [np.asarray(saddle_ratio, dtype=float), np.asarray(validation_ratio)]
    )
    order = np.argsort(combined_bandwidth)
    combined_bandwidth = combined_bandwidth[order]
    combined_ratio = combined_ratio[order]
    unique_bandwidth, unique_indices = np.unique(
        combined_bandwidth, return_index=True
    )
    calibrated_ratio = combined_ratio[unique_indices]
    return {
        "time_bandwidth": time_bandwidth,
        "saddle_ratio": np.asarray(saddle_ratio),
        "calibrated_bandwidth": unique_bandwidth,
        "calibrated_ratio": calibrated_ratio,
        "validation_bandwidth": validation_bandwidth,
        "validation_ratio": np.asarray(validation_ratio),
        "validation_error": np.asarray(validation_error),
        "validation_tail_ratio": np.asarray(validation_tail_ratio),
        "validation_tail_error": np.asarray(validation_tail_error),
        "validation_detection_ratio": np.asarray(validation_detection_ratio),
        "validation_detection_error": np.asarray(validation_detection_error),
        "maximum_tail_relative_error": float(
            np.max(np.abs(np.asarray(validation_tail_ratio) - 1.0))
        ),
        "maximum_detection_relative_error": float(
            np.max(np.abs(np.asarray(validation_detection_ratio) - 1.0))
        ),
    }


def finite_sample_field_correction(freq: np.ndarray) -> np.ndarray:
    """Interpolate the calibrated field correction for the fixed scan."""
    frequencies = np.asarray(freq, dtype=float)
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("Correction frequencies must be finite and positive")
    calibration = finite_sample_calibration_curve()
    time_bandwidth = T_PROJECTION * frequencies * V0_DM**2 / 2.0
    return np.interp(
        np.log(time_bandwidth),
        np.log(calibration["calibrated_bandwidth"]),
        calibration["calibrated_ratio"],
        left=unresolved_exact_to_asimov_amplitude_factor(T_PROJECTION),
        right=1.0,
    )


def axion_mass_ev(freq: np.ndarray) -> np.ndarray:
    """Rest-mass energy in eV for a cycle frequency in hertz."""
    return PLANCK_EV_S * np.asarray(freq, dtype=float)


def axion_electron_coupling(field_rms_ft: np.ndarray) -> np.ndarray:
    """Map electron-equivalent rms field to g_ae for the halo model."""
    rho_ev4 = RHO_DM_GEV_CM3 * 1.0e9 / CM_IN_EV_INV**3
    v_parallel_rms = V0_DM / math.sqrt(2.0)
    field_t = np.asarray(field_rms_ft, dtype=float) * 1.0e-15
    return (
        field_t
        * M_E_EV
        * ELECTRON_G_FACTOR
        * MU_B_EV_T
        / (math.sqrt(rho_ev4) * v_parallel_rms)
    )


def electron_equivalent_field_ft(coupling: np.ndarray) -> np.ndarray:
    """Inverse of axion_electron_coupling for the round-trip check."""
    rho_ev4 = RHO_DM_GEV_CM3 * 1.0e9 / CM_IN_EV_INV**3
    v_parallel_rms = V0_DM / math.sqrt(2.0)
    return (
        np.asarray(coupling, dtype=float)
        * math.sqrt(rho_ev4)
        * v_parallel_rms
        / (M_E_EV * ELECTRON_G_FACTOR * MU_B_EV_T)
        / 1.0e-15
    )


def model_noise(freq: np.ndarray, variance: float) -> dict[str, np.ndarray]:
    """Synthetic port-resolved noise model in fT/sqrt(Hz).

    Shot and laser terms are output-referred after low-frequency field
    calibration by |H_B(0)|; atomic, environmental, and light-shift terms are
    input-equivalent fields. Backaction and correlations enter separately
    through the stress extensions.
    """
    freq = np.asarray(freq, dtype=float)
    if freq.ndim != 1 or np.any(~np.isfinite(freq)) or np.any(freq <= 0.0):
        raise ValueError(
            "Noise frequencies must be a finite positive one-dimensional array"
        )
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("Detected variance must be finite and nonnegative")
    shot_out = NOISE_LEVELS["shot_out"] * np.sqrt(1.0 + (freq / 520.0) ** 2)
    laser_out = NOISE_LEVELS["laser_out"] * np.sqrt(1.0 + (2.6 / freq) ** 2)
    atom_in = np.full_like(freq, NOISE_LEVELS["atom_in"])
    env_in = NOISE_LEVELS["environment_in"] * np.sqrt(1.0 + (0.075 / freq) ** 2)
    light_shift_in = NOISE_LEVELS["light_shift_in"] * np.sqrt(
        1.0 + (0.45 / freq) ** 2
    )
    # First-order response |H_B(2 pi f)|/|H_B(0)| for the representative
    # transverse relaxation rate; the static calibration factor is absorbed
    # into the output-referred amplitudes.
    response = np.abs(normalized_transfer(freq))
    components = {
        "shot_out": shot_out,
        "laser_out": laser_out,
        "atom_in": atom_in,
        "environment_in": env_in,
        "light_shift_in": light_shift_in,
        "shot": shot_out / response,
        "laser": laser_out / response,
        "atom": atom_in,
        "environment": env_in,
        "light_shift": light_shift_in,
    }
    imprecision2 = (shot_out**2 + laser_out**2) / response**2
    input2 = atom_in**2 + env_in**2 + light_shift_in**2
    components["total_coherent"] = np.sqrt(imprecision2 + input2)
    components["total_squeezed"] = np.sqrt(
        (variance * shot_out**2 + laser_out**2) / response**2 + input2
    )
    components["response"] = response
    return components


@functools.lru_cache(maxsize=1)
def shared_gain_normalization() -> Normalize:
    """Joint display scale for the Fig. 1(c) and Fig. 2(a,b) gain grids."""
    reference = model_noise(np.asarray([10.0]), 1.0)
    shot2 = float(reference["shot"][0] ** 2)
    other2 = float(reference["total_coherent"][0] ** 2 - shot2)
    grids = (
        (
            np.asarray([R_FID]),
            np.linspace(0.5, 1.0, 180),
            np.linspace(0.0, 0.16, 180),
        ),
        (
            np.linspace(0.0, 3.2, 260),
            np.linspace(0.45, 1.0, 160),
            np.asarray([PHASE_FID]),
        ),
        (
            np.linspace(0.0, 3.2, 260),
            np.asarray([ETA_FID]),
            np.linspace(0.0, 0.20, 180),
        ),
    )
    maximum = max(
        math.sqrt(
            (shot2 + other2)
            / (
                effective_variance(float(rr), float(eta), float(ph)) * shot2
                + other2
            )
        )
        for rvals, etavals, phivals in grids
        for rr in rvals
        for eta in etavals
        for ph in phivals
    )
    return Normalize(vmin=0.0, vmax=math.ceil(5.0 * maximum) / 5.0)


def set_shared_10hz_gain_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0.0, 2.2)
    ax.set_yticks(SHARED_GAIN_TICKS)


def make_fig1(freq: np.ndarray, noise: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    a, b, c, d = ax.flat
    response = np.abs(normalized_transfer(freq))
    a.loglog(freq, response, color=BLUE, lw=1.2)
    a.axvline(135.0, color=GRAY, ls="--", lw=0.8)
    a.set(
        xlabel="Frequency (Hz)",
        ylabel="Normalized response",
        xlim=(freq.min(), freq.max()),
    )
    label_panel(a, "(a)")

    colors = {
        "shot": BLUE,
        "atom": GREEN,
        "laser": ORANGE,
        "environment": MAGENTA,
        "light_shift": GOLD,
    }
    labels = {
        "shot": "optical shot",
        "atom": "atomic process",
        "laser": "laser readout",
        "environment": "environment",
        "light_shift": "light shift",
    }
    for key in ("shot", "atom", "laser", "environment", "light_shift"):
        b.loglog(freq, noise[key], color=colors[key], lw=0.9, label=labels[key])
    b.loglog(freq, noise["total_coherent"], color=BLACK, lw=1.4, label="total")
    b.set(xlabel="Frequency (Hz)", ylabel=r"ASD (fT $\cdot$ Hz$^{-1/2}$)")
    b.legend(
        ncol=2,
        loc="upper left",
        handlelength=1.4,
        columnspacing=0.8,
    )
    b.set_ylim(top=float(np.max(noise["total_coherent"])) * 3.0)
    label_panel(b, "(b)")

    # Gain grid at 10 Hz, where the fiducial model is close to optical-noise
    # limited.
    eta_grid = np.linspace(0.5, 1.0, 180)
    phase_grid = np.linspace(0.0, 0.16, 180)
    gain = np.empty((len(phase_grid), len(eta_grid)))
    reference = model_noise(np.asarray([10.0]), 1.0)
    shot2 = float(reference["shot"][0] ** 2)
    other2 = float(reference["total_coherent"][0] ** 2 - shot2)
    for i, ph in enumerate(phase_grid):
        for j, et in enumerate(eta_grid):
            vv = effective_variance(R_FID, float(et), float(ph))
            gain[i, j] = math.sqrt((shot2 + other2) / (vv * shot2 + other2))
    im = c.pcolormesh(
        eta_grid,
        phase_grid,
        gain,
        shading="auto",
        cmap="cividis",
        rasterized=True,
        edgecolors="none",
        linewidth=0.0,
        norm=shared_gain_normalization(),
    )
    contour_set = c.contour(
        eta_grid,
        phase_grid,
        gain,
        levels=[1.05, 1.10, 1.25, 1.50],
        colors="white",
        linewidths=0.7,
    )
    c.clabel(contour_set, fmt="%.2f", colors="white", fontsize=7, inline=True)
    c.set(xlabel="Detection efficiency", ylabel="Phase jitter (rad)")
    fig.colorbar(
        im,
        ax=c,
        pad=0.01,
        fraction=0.045,
        shrink=0.96,
        aspect=28,
        label="Amplitude gain at 10 Hz",
        ticks=SHARED_GAIN_TICKS,
    )
    label_panel(c, "(c)")

    coherent = noise["total_coherent"]
    squeezed = noise["total_squeezed"]
    d.semilogx(freq, coherent / squeezed, color=BLUE, lw=1.2)
    d.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    d.set(xlabel="Frequency (Hz)", ylabel="Gain from squeezing")
    d.set_ylim(0.8, 5.0)
    d.set_yticks([1, 2, 3, 4, 5])
    label_panel(d, "(d)")
    save_figure(fig, "fig1_noise_budget")


def make_fig2(freq: np.ndarray, noise: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    a, b, c, d = ax.flat
    rvals = np.linspace(0.0, 3.2, 260)
    etavals = np.linspace(0.45, 1.0, 160)
    rg, eg = np.meshgrid(rvals, etavals)
    v = np.vectorize(
        lambda rr, ee: effective_variance(float(rr), float(ee), PHASE_FID)
    )(rg, eg)
    reference = model_noise(np.asarray([10.0]), 1.0)
    shot2 = float(reference["shot"][0] ** 2)
    other2 = float(reference["total_coherent"][0] ** 2 - shot2)
    gain = np.sqrt((shot2 + other2) / (v * shot2 + other2))

    phivals = np.linspace(0.0, 0.20, 180)
    r2, p2 = np.meshgrid(rvals, phivals)
    vv = np.vectorize(
        lambda rr, pp: effective_variance(float(rr), ETA_FID, float(pp))
    )(r2, p2)
    gain2 = np.sqrt((shot2 + other2) / (vv * shot2 + other2))
    im = a.pcolormesh(
        rvals,
        etavals,
        gain,
        shading="auto",
        cmap="cividis",
        rasterized=True,
        edgecolors="none",
        linewidth=0.0,
        norm=shared_gain_normalization(),
    )
    contours_a = a.contour(
        rvals,
        etavals,
        gain,
        levels=[1.5, 1.75],
        colors="white",
        linewidths=0.7,
    )
    a.clabel(contours_a, fmt="%.1f", colors="white", fontsize=7, inline=True)
    b.pcolormesh(
        rvals,
        phivals,
        gain2,
        shading="auto",
        cmap="cividis",
        rasterized=True,
        edgecolors="none",
        linewidth=0.0,
        norm=shared_gain_normalization(),
    )
    contours_b = b.contour(
        rvals,
        phivals,
        gain2,
        levels=[1.5, 1.75],
        colors="white",
        linewidths=0.7,
    )
    b.clabel(contours_b, fmt="%.1f", colors="white", fontsize=7, inline=True)
    a.set(xlabel="Squeezing parameter $r$", ylabel="Detection efficiency $\eta$")
    b.set(xlabel="Squeezing parameter $r$", ylabel=r"Phase jitter $\sigma_\theta$ (rad)")
    fig.colorbar(
        im,
        ax=b,
        pad=0.01,
        fraction=0.045,
        shrink=0.96,
        aspect=28,
        label="Amplitude gain at 10 Hz",
        ticks=SHARED_GAIN_TICKS,
    )
    label_panel(a, "(a)")
    label_panel(b, "(b)")

    for ph, col, lab in [
        (0.0, BLUE, "0 rad"),
        (PHASE_FID, ORANGE, f"{PHASE_FID:.3f} rad"),
        (0.10, GREEN, "0.10 rad"),
    ]:
        vals = [effective_variance(float(rr), ETA_FID, ph) for rr in rvals]
        gg = np.sqrt((shot2 + other2) / (np.asarray(vals) * shot2 + other2))
        c.plot(10 * np.log10(np.exp(2 * rvals)), gg, color=col, lw=1.2, label=lab)
    c.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    c.set(xlabel="Source squeezing (dB)", ylabel="Gain at 10 Hz")
    set_shared_10hz_gain_axis(c)
    c.legend(loc="lower left")
    label_panel(c, "(c)")

    # The analytic optimum is frequency independent here; the maximum gain is
    # not, because the OPM response weights the two ports differently.
    freqs = np.logspace(-2, 3.2, 240)
    r_opt = optimum_squeezing_r(PHASE_FID)
    opts = np.full_like(freqs, r_opt)
    optg = []
    for ff in freqs:
        nn = model_noise(np.asarray([ff]), 1.0)
        sh2 = float(nn["shot"][0] ** 2)
        oth2 = float(nn["total_coherent"][0] ** 2 - sh2)
        vv0 = effective_variance(r_opt, ETA_FID, PHASE_FID)
        optg.append(math.sqrt((sh2 + oth2) / (vv0 * sh2 + oth2)))
    d.semilogx(freqs, opts, color=BLUE, lw=1.2, label="covariance optimum")
    d2 = d.twinx()
    d2.semilogx(freqs, optg, color=ORANGE, ls="--", lw=0.9, label="total gain")
    d.set(xlabel="Frequency (Hz)", ylabel="Optimal squeezing $r$")
    d2.set_ylabel("Maximum total gain", color=ORANGE)
    d.tick_params(axis="y", colors=BLUE)
    d2.tick_params(axis="y", colors=ORANGE)
    d.yaxis.label.set_color(BLUE)
    d2.yaxis.label.set_color(ORANGE)
    handles1, labels1 = d.get_legend_handles_labels()
    handles2, labels2 = d2.get_legend_handles_labels()
    d2.set_ylim(0.8, 5.0)
    d2.set_yticks([1, 2, 3, 4, 5])
    d2.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    d.legend(handles1 + handles2, labels1 + labels2, loc="upper left", handlelength=1.4)
    label_panel(d, "(d)")
    save_figure(fig, "fig2_loss_phase")


def make_fig3(rng: np.random.Generator) -> dict[str, float]:
    fig, ax = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    a, b, c, d = ax.flat
    n = 2**15
    dt = 0.02
    f0 = 12.3
    f = np.fft.rfftfreq(n, dt)
    # Synthetic periodograms with a narrow coherent line, fixed random draw.
    base = model_noise(np.maximum(f, 1e-3), 1.0)["total_coherent"]
    vf = effective_variance(R_FID, ETA_FID, PHASE_FID)
    sq = model_noise(np.maximum(f, 1e-3), vf)["total_squeezed"]
    coh_period = base**2 * rng.gamma(shape=1.0, scale=1.0, size=f.size)
    sq_period = sq**2 * rng.gamma(shape=1.0, scale=1.0, size=f.size)
    line = 8.0 * np.exp(-0.5 * ((f - f0) / 0.035) ** 2)
    coh_period += line
    sq_period += line
    window = (f > 8.0) & (f < 16.0)
    a.loglog(
        f[1:], np.sqrt(coh_period[1:]), color=BLUE, lw=0.6, alpha=0.85, label="coherent probe"
    )
    a.loglog(
        f[1:], np.sqrt(sq_period[1:]), color=ORANGE, lw=0.6, alpha=0.85, label="squeezed probe"
    )
    a.set(xlabel="Frequency (Hz)", ylabel=r"ASD (fT $\cdot$ Hz$^{-1/2}$)")
    # Independent synthetic FFT grid; Nyquist at 25 Hz for this sampling.
    a.set_xlim(max(0.1, float(f[1])), float(f[-1]))
    a.legend(
        loc="lower left",
    )
    label_panel(a, "(a)")

    b.plot(f[window], np.sqrt(coh_period[window]), color=BLUE, lw=0.6, label="coherent")
    b.plot(f[window], np.sqrt(sq_period[window]), color=ORANGE, lw=0.6, label="squeezed")
    b.axvline(f0, color=BLACK, ls=":", lw=0.7)
    b.set(xlabel="Frequency (Hz)", ylabel=r"ASD (fT $\cdot$ Hz$^{-1/2}$)", xlim=(8.0, 16.0))
    b.set_ylim(
        0.0,
        1.12
        * float(np.sqrt(max(np.max(coh_period[window]), np.max(sq_period[window])))),
    )
    b.legend(loc="upper left")
    label_panel(b, "(b)")

    # Deterministic-tone noncentral amplitude versus integration time.
    times = np.logspace(0, 5, 100)
    n0 = float(model_noise(np.asarray([f0]), 1.0)["total_coherent"][0])
    ns = float(model_noise(np.asarray([f0]), vf)["total_squeezed"][0])
    amp = 2.1
    # Peak sinusoidal amplitude; sqrt(lambda) = A_peak sqrt(T/S1) one-sided.
    snr_c = amp * np.sqrt(times) / n0
    snr_s = amp * np.sqrt(times) / ns
    c.loglog(times, snr_c, color=BLUE, lw=1.2, label="coherent probe")
    c.loglog(times, snr_s, color=ORANGE, lw=1.2, label="squeezed probe")
    coefficient_peak = math.sqrt(required_noncentrality())
    c.axhline(
        coefficient_peak, color=BLACK, ls=":", lw=0.7, label="50% detection threshold"
    )
    c.text(
        0.98,
        coefficient_peak * 1.08,
        f"{coefficient_peak:.2f}",
        transform=c.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color=GRAY,
        fontsize=8,
        fontfamily="Arial",
    )
    c.set(xlabel="Coherent integration time (s)", ylabel="Noncentral amplitude")
    c.legend(loc="upper left", fontsize=7.3, handlelength=1.4)
    label_panel(c, "(c)")

    # Distribution of the normalized periodogram ratio away from the line.
    ratio_window = window & (np.abs(f - f0) > 0.2)
    ratio = coh_period[ratio_window] / sq_period[ratio_window]
    ratio_bins = np.logspace(np.log10(np.min(ratio)), np.log10(np.max(ratio)), 33)
    d.hist(ratio, bins=ratio_bins, density=True, color=PALE_BLUE, edgecolor="none", alpha=0.45)
    d.axvline(float(np.median(ratio)), color=ORANGE, lw=1.2, label="median")
    d.set(xlabel="Power ratio (coherent / squeezed)", ylabel="Density", ylim=(0.0, None))
    d.set_xscale("log")
    d.legend(loc="best")
    label_panel(d, "(d)")
    save_figure(fig, "fig3_synthetic_search")
    return {
        "f0": f0,
        "median_periodogram_ratio": float(np.median(ratio)),
        "coherent_snr_1ks": float(amp * np.sqrt(1000.0) / n0),
        "squeezed_snr_1ks": float(amp * np.sqrt(1000.0) / ns),
    }


def make_fig4(freq: np.ndarray, noise: dict[str, np.ndarray], vfid: float) -> dict[str, float]:
    fig, ax = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    a, b, c, d = ax.flat
    coh = noise["total_coherent"]
    sq = noise["total_squeezed"]
    field_coh_det = deterministic_asimov_target(coh, T_PROJECTION)
    field_sq_det = deterministic_asimov_target(sq, T_PROJECTION)
    field_coh_uldm = ul_dm_asimov_target(coh, freq, T_PROJECTION)
    field_sq_uldm = ul_dm_asimov_target(sq, freq, T_PROJECTION)
    finite_correction = finite_sample_field_correction(freq)
    field_coh_calibrated = field_coh_uldm * finite_correction
    field_sq_calibrated = field_sq_uldm * finite_correction
    a.loglog(freq, field_coh_uldm, color=BLUE, lw=1.2)
    a.loglog(freq, field_sq_uldm, color=ORANGE, lw=1.2)
    a.loglog(freq, field_coh_det, color=BLUE, ls="--", lw=0.8)
    a.loglog(freq, field_sq_det, color=ORANGE, ls="--", lw=0.8)
    a.loglog(freq, field_coh_calibrated, color=BLUE, ls=":", lw=0.8)
    a.loglog(freq, field_sq_calibrated, color=ORANGE, ls=":", lw=0.8)
    a.set(xlabel="Rest-mass frequency (Hz)", ylabel="ULDM target field")
    legend_handles = [
        Line2D([], [], color=BLUE, lw=1.2, label="coherent probe"),
        Line2D([], [], color=ORANGE, lw=1.2, label="squeezed probe"),
        Line2D([], [], color=BLACK, lw=1.2, ls="-", label="Asimov"),
        Line2D([], [], color=BLACK, lw=0.8, ls="--", label="deterministic tone"),
        Line2D([], [], color=BLACK, lw=0.8, ls=":", label="50% median power"),
    ]
    a.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper left",
        ncol=1,
        fontsize=7.5,
        handlelength=1.4,
        labelspacing=0.25,
    )
    label_panel(a, "(a)")

    mass = axion_mass_ev(freq)
    coupling_coh = axion_electron_coupling(field_coh_uldm)
    coupling_sq = axion_electron_coupling(field_sq_uldm)
    coupling_coh_calibrated = axion_electron_coupling(field_coh_calibrated)
    coupling_sq_calibrated = axion_electron_coupling(field_sq_calibrated)
    b.loglog(mass, coupling_coh, color=BLUE, lw=1.2, label="coherent probe")
    b.loglog(mass, coupling_sq, color=ORANGE, lw=1.2, label="squeezed probe")
    b.loglog(mass, coupling_coh_calibrated, color=BLUE, ls=":", lw=0.8)
    b.loglog(mass, coupling_sq_calibrated, color=ORANGE, ls=":", lw=0.8)
    b.set(xlabel=r"Axion mass $m_a$ (eV/$c^2$)", ylabel=r"ULDM target $g_{ae}$")
    b.set_ylim(top=6.0 * float(max(np.max(coupling_coh), np.max(coupling_coh_calibrated))))
    b.legend(loc="upper left", ncol=2, fontsize=7.5, handlelength=1.2, columnspacing=0.7)
    label_panel(b, "(b)")

    # Smooth stochastic-line factor from the integrated Maxwellian template.
    for obs_time, color, label in [
        (1.0e3, BLUE, r"$T=10^3$ s"),
        (T_PROJECTION, GREEN, r"$T=10^5$ s"),
        (1.0e6, ORANGE, r"$T=10^6$ s"),
    ]:
        c.semilogx(
            freq, stochastic_line_factor(freq, obs_time), color=color, lw=1.2, label=label
        )
    c.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    c.set(xlabel="Rest-mass frequency (Hz)", ylabel="Stochastic-line factor")
    c.legend(loc="upper left")
    label_panel(c, "(c)")

    gain = field_coh_uldm / field_sq_uldm
    d.semilogx(freq, gain, color=GREEN, lw=1.2)
    d.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    d.set(xlabel="Signal frequency (Hz)", ylabel="Target improvement")
    d.set_ylim(0.8, 5.0)
    d.set_yticks([1, 2, 3, 4, 5])
    label_panel(d, "(d)")
    save_figure(fig, "fig4_sensitivity")
    eval_freq = np.asarray([10.0, 300.0])
    eval_coh = model_noise(eval_freq, 1.0)["total_coherent"]
    eval_sq = model_noise(eval_freq, vfid)["total_squeezed"]
    eval_coh_uldm = ul_dm_asimov_target(eval_coh, eval_freq, T_PROJECTION)
    eval_sq_uldm = ul_dm_asimov_target(eval_sq, eval_freq, T_PROJECTION)
    eval_correction = finite_sample_field_correction(eval_freq)
    return {
        "field_target_coherent_10hz": float(eval_coh_uldm[0]),
        "field_target_squeezed_10hz": float(eval_sq_uldm[0]),
        "field_target_coherent_calibrated_10hz": float(
            eval_coh_uldm[0] * eval_correction[0]
        ),
        "field_target_squeezed_calibrated_10hz": float(
            eval_sq_uldm[0] * eval_correction[0]
        ),
        "field_target_coherent_deterministic_10hz": float(
            deterministic_asimov_target(eval_coh, T_PROJECTION)[0]
        ),
        "field_target_squeezed_deterministic_10hz": float(
            deterministic_asimov_target(eval_sq, T_PROJECTION)[0]
        ),
        "gain_10hz": float(eval_coh_uldm[0] / eval_sq_uldm[0]),
        "gain_300hz": float(eval_coh_uldm[1] / eval_sq_uldm[1]),
        "gain_10hz_deterministic": float(
            deterministic_asimov_target(eval_coh, T_PROJECTION)[0]
            / deterministic_asimov_target(eval_sq, T_PROJECTION)[0]
        ),
        "gain_300hz_deterministic": float(
            deterministic_asimov_target(eval_coh, T_PROJECTION)[1]
            / deterministic_asimov_target(eval_sq, T_PROJECTION)[1]
        ),
        "coherence_time_10hz_s": float(coherence_time(np.asarray([10.0]))[0]),
        "coherence_time_300hz_s": float(coherence_time(np.asarray([300.0]))[0]),
        "line_bins_10hz": int(line_bin_count(np.asarray([10.0]), T_PROJECTION)[0]),
        "line_bins_300hz": int(line_bin_count(np.asarray([300.0]), T_PROJECTION)[0]),
        "stochastic_line_factor_10hz": float(
            stochastic_line_factor(np.asarray([10.0]), T_PROJECTION)[0]
        ),
        "stochastic_line_factor_300hz": float(
            stochastic_line_factor(np.asarray([300.0]), T_PROJECTION)[0]
        ),
        "axion_coupling_coherent_10hz": float(
            axion_electron_coupling(eval_coh_uldm)[0]
        ),
        "axion_coupling_squeezed_10hz": float(
            axion_electron_coupling(eval_sq_uldm)[0]
        ),
        "axion_coupling_coherent_calibrated_10hz": float(
            axion_electron_coupling(eval_coh_uldm * eval_correction)[0]
        ),
        "axion_coupling_squeezed_calibrated_10hz": float(
            axion_electron_coupling(eval_sq_uldm * eval_correction)[0]
        ),
        "finite_sample_correction_10hz": float(eval_correction[0]),
    }


def make_fig5(freq: np.ndarray, noise: dict[str, np.ndarray]) -> dict[str, float]:
    fig, ax = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    a, b, c, d = ax.flat
    rr = np.linspace(0.0, 3.2, 520)
    reference = model_noise(np.asarray([10.0]), 1.0)
    sh2 = float(reference["shot"][0] ** 2)
    oth2 = float(reference["total_coherent"][0] ** 2 - sh2)
    eta_colors = [(0.65, BLUE), (0.80, GREEN), (0.90, MAGENTA), (0.97, ORANGE)]
    for eta, color in eta_colors:
        gains = [
            math.sqrt(
                (sh2 + oth2)
                / (effective_variance(float(r), eta, PHASE_FID) * sh2 + oth2)
            )
            for r in rr
        ]
        a.plot(
            10.0 * np.log10(np.exp(2.0 * rr)),
            gains,
            color=color,
            lw=1.2,
            label=fr"$\eta={eta:.2f}$",
        )
    a.set(xlabel="Source squeezing (dB)", ylabel="Gain at 10 Hz")
    set_shared_10hz_gain_axis(a)
    a.legend(
        ncol=2,
        loc="lower right",
        handlelength=1.4,
        columnspacing=0.8,
    )
    label_panel(a, "(a)")

    # Gain versus phase jitter at fixed source squeezing.
    ph = np.linspace(0.0, 0.25, 300)
    fid_db = 10.0 * np.log10(np.exp(2.0 * R_FID))
    source_levels = [
        (0.35, BLUE, "3 dB"),
        (1.80, MAGENTA, "15.6 dB"),
        (R_FID, ORANGE, f"{fid_db:.1f} dB"),
    ]
    for r, color, label in source_levels:
        gains = [
            math.sqrt(
                (sh2 + oth2)
                / (effective_variance(r, ETA_FID, float(p)) * sh2 + oth2)
            )
            for p in ph
        ]
        b.plot(ph, gains, color=color, lw=1.2, label=label)
    b.set(xlabel="Phase jitter (rad)", ylabel="Gain at 10 Hz")
    set_shared_10hz_gain_axis(b)
    b.legend(loc="upper right")
    label_panel(b, "(b)")

    # Optical shot-noise fraction controls the idealized gain.
    fractions = np.linspace(0.0, 1.0, 250)
    for r, color, label in source_levels:
        vv = effective_variance(r, ETA_FID, PHASE_FID)
        gains = np.sqrt(1.0 / (fractions * vv + (1.0 - fractions)))
        c.plot(fractions, gains, color=color, lw=1.2, label=label)
    c.set(xlabel="Optical shot-noise fraction", ylabel="Idealized gain")
    c.legend(loc="upper left")
    label_panel(c, "(c)")

    # Effective variance under Gaussian phase noise.
    for r, color, label in source_levels:
        vals = [effective_variance(r, ETA_FID, float(p)) for p in ph]
        d.plot(ph, vals, color=color, lw=1.2, label=label)
    d.set(xlabel="Phase jitter (rad)", ylabel="Detected quadrature variance")
    d.set_yscale("log")
    d.legend(loc="lower right")
    label_panel(d, "(d)")
    save_figure(fig, "fig5_optimization")
    r_opt = optimum_squeezing_r(PHASE_FID)
    gain_opt = math.sqrt(
        (sh2 + oth2)
        / (effective_variance(r_opt, ETA_FID, PHASE_FID) * sh2 + oth2)
    )
    return {
        "optimal_source_db_10hz": float(10.0 * np.log10(np.exp(2.0 * r_opt))),
        "optimal_gain_10hz": float(gain_opt),
    }


def monte_carlo_statistics(
    rng: np.random.Generator, nmc: int = 4000
) -> dict[str, np.ndarray]:
    """Draw the design-stress prior, including omitted-channel tests."""
    eta = rng.uniform(0.50, 0.99, nmc)
    r = rng.uniform(0.0, 2.8, nmc)
    phase = np.exp(rng.uniform(np.log(1.0e-3), np.log(0.30), nmc))
    shot_fraction = rng.uniform(0.0, 1.0, nmc)
    epsilon_ba = np.exp(
        rng.uniform(np.log(1.0e-4), np.log(BACKACTION_STRESS_MAX), nmc)
    )
    correlation_magnitude = rng.uniform(0.0, CORRELATION_STRESS_MAX, nmc)
    correlation_phase = rng.uniform(-np.pi, np.pi, nmc)
    vv = np.array(
        [
            effective_variance(float(rr), float(ee), float(pp))
            for rr, ee, pp in zip(r, eta, phase)
        ]
    )
    reference_variance = np.array(
        [
            effective_variance(R_FID, float(ee), float(pp))
            for ee, pp in zip(eta, phase)
        ]
    )
    reference_psd = shot_fraction * reference_variance + 1.0 - shot_fraction
    excess_ba = (
        epsilon_ba
        * reference_psd
        * (np.exp(2.0 * r) - 1.0)
        / (math.exp(2.0 * R_FID) - 1.0)
    )
    # Ten percent of the nonoptical fraction goes to the light-shift channel
    # for the bounded covariance stress.
    light_shift_fraction = 0.10 * (1.0 - shot_fraction)
    response_phase = float(np.angle(normalized_transfer(np.asarray([10.0]))[0]))
    phase_projection = np.cos(correlation_phase - response_phase)
    coherent_cross = (
        2.0
        * correlation_magnitude
        * np.sqrt(shot_fraction * light_shift_fraction)
        * phase_projection
    )
    squeezed_cross = (
        2.0
        * correlation_magnitude
        * np.sqrt(np.maximum(0.0, shot_fraction * vv * light_shift_fraction))
        * phase_projection
    )
    coherent_psd = 1.0 + coherent_cross
    squeezed_psd = (
        shot_fraction * vv
        + 1.0
        - shot_fraction
        + excess_ba
        + squeezed_cross
    )
    if np.any(coherent_psd <= 0.0) or np.any(squeezed_psd <= 0.0):
        raise RuntimeError("The robustness covariance produced a nonpositive PSD")
    gains = np.sqrt(coherent_psd / squeezed_psd)
    return {
        "eta": eta,
        "r": r,
        "phase": phase,
        "shot_fraction": shot_fraction,
        "epsilon_ba": epsilon_ba,
        "correlation_magnitude": correlation_magnitude,
        "correlation_phase": correlation_phase,
        "gains": gains,
    }


def make_fig6(
    rng: np.random.Generator, freq: np.ndarray, noise: dict[str, np.ndarray]
) -> dict[str, float]:
    fig, ax = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    a, b, c, d = ax.flat
    nmc = 4000
    # Design-prior draws. The ranges include readout-poor cases on purpose.
    draws = monte_carlo_statistics(rng, nmc)
    epsilon_ba = draws["epsilon_ba"]
    correlation_magnitude = draws["correlation_magnitude"]
    shot_fraction = draws["shot_fraction"]
    gains = draws["gains"]
    a.hist(gains, bins=45, density=True, color=PALE_BLUE, edgecolor="white", alpha=0.45)
    a.axvline(np.median(gains), color=ORANGE, lw=1.2, label="median")
    a.axvline(
        np.percentile(gains, 5), color=BLACK, ls=":", lw=0.7, label="5th percentile"
    )
    a.set(xlabel="Total sensitivity gain", ylabel="Density", ylim=(0.0, None))
    a.legend(loc="upper right")
    label_panel(a, "(a)")

    b.scatter(epsilon_ba, gains, s=9, alpha=0.18, color=BLUE, edgecolors="none")
    b.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    b.set(xlabel=r"Backaction fraction $\epsilon_{\rm BA}$", ylabel="Total gain")
    b.set_xscale("log")
    label_panel(b, "(b)")

    c.scatter(
        correlation_magnitude, gains, s=9, alpha=0.18, color=ORANGE, edgecolors="none"
    )
    c.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    c.set(xlabel=r"Correlation magnitude $\rho$", ylabel="Total gain")
    label_panel(c, "(c)")

    d.scatter(shot_fraction, gains, s=9, alpha=0.18, color=GREEN, edgecolors="none")
    d.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    d.set(xlabel="Optical shot-noise fraction", ylabel="Total gain")
    upper_gain = max(5.0, math.ceil(float(np.max(gains)) * 1.05))
    for scatter_ax in (b, c, d):
        scatter_ax.set_ylim(0.0, upper_gain)
        scatter_ax.set_yticks(np.arange(0.0, upper_gain + 0.5, 1.0))
    label_panel(d, "(d)")
    save_figure(fig, "fig6_robustness")
    return {
        "mc_median_gain": float(np.median(gains)),
        "mc_p05_gain": float(np.percentile(gains, 5)),
        "mc_p95_gain": float(np.percentile(gains, 95)),
        "mc_fraction_above_1p1": float(np.mean(gains > 1.10)),
    }


def make_fig7(freq: np.ndarray, noise: dict[str, np.ndarray]) -> dict[str, float]:
    """Phenomenological validity envelope for backaction and correlations."""
    fig, ax = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    a, b, c, d = ax.flat
    reference = model_noise(np.asarray([10.0]), 1.0)
    coherent_psd = float(reference["total_coherent"][0] ** 2)
    shot_psd = float(reference["shot"][0] ** 2)
    optical_fraction = shot_psd / coherent_psd
    r_values = np.linspace(0.0, 3.2, 420)
    epsilon_levels = [0.0, 0.01, 0.10, 0.30, 1.00]
    colors = [BLUE, GREEN, GOLD, ORANGE, MAGENTA]
    for epsilon_ba, color in zip(epsilon_levels, colors):
        gains = [
            stressed_gain(float(rr), epsilon_ba, optical_fraction)
            for rr in r_values
        ]
        a.plot(
            10.0 * np.log10(np.exp(2.0 * r_values)),
            gains,
            color=color,
            lw=1.2,
            label=fr"$\epsilon_{{\rm BA}}={epsilon_ba:g}$",
        )
    a.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    a.set(xlabel="Source squeezing (dB)", ylabel="Gain at 10 Hz")
    set_shared_10hz_gain_axis(a)
    a.legend(
        ncol=3,
        loc="lower left",
        fontsize=7.5,
        handlelength=1.4,
        columnspacing=0.8,
    )
    label_panel(a, "(a)")

    epsilon_grid = np.logspace(-4.0, 0.0, 260)
    optimum_r = np.asarray(
        [stressed_optimum_r(float(ee), optical_fraction) for ee in epsilon_grid]
    )
    optimum_db = 10.0 * np.log10(np.exp(2.0 * optimum_r))
    b.semilogx(epsilon_grid, optimum_db, color=BLUE, lw=1.2)
    b.axhline(10.0 * np.log10(np.exp(2.0 * R_FID)), color=GRAY, ls="--", lw=0.8)
    b.set(
        xlabel=r"Backaction fraction $\epsilon_{\rm BA}$",
        ylabel="Optimal source squeezing (dB)",
    )
    label_panel(b, "(b)")

    optimized_gain = np.asarray(
        [
            stressed_gain(float(rr), float(ee), optical_fraction)
            for rr, ee in zip(optimum_r, epsilon_grid)
        ]
    )
    fixed_gain = np.asarray(
        [stressed_gain(R_FID, float(ee), optical_fraction) for ee in epsilon_grid]
    )
    c.semilogx(epsilon_grid, optimized_gain, color=BLUE, lw=1.2, label="reoptimized")
    c.semilogx(
        epsilon_grid, fixed_gain, color=ORANGE, ls="--", lw=0.9, label="fixed at 20.97 dB"
    )
    c.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    c.set(xlabel=r"Backaction fraction $\epsilon_{\rm BA}$", ylabel="Gain at 10 Hz")
    set_shared_10hz_gain_axis(c)
    c.legend(loc="lower left")
    label_panel(c, "(c)")

    rho_grid = np.linspace(0.0, CORRELATION_STRESS_MAX, 140)
    phase_grid = np.linspace(-np.pi, np.pi, 220)
    rrho, pphase = np.meshgrid(rho_grid, phase_grid, indexing="ij")
    stressed_r = stressed_optimum_r(0.10, optical_fraction)
    correlated_gain = np.empty_like(rrho)
    for i in range(rrho.shape[0]):
        for j in range(rrho.shape[1]):
            numerator = correlated_total_psd(
                10.0, 0.0, float(rrho[i, j]), float(pphase[i, j])
            )
            denominator = correlated_total_psd(
                10.0,
                stressed_r,
                float(rrho[i, j]),
                float(pphase[i, j]),
                epsilon_ba=0.10,
            )
            correlated_gain[i, j] = math.sqrt(numerator / denominator)
    image_map = d.pcolormesh(
        phase_grid,
        rho_grid,
        correlated_gain,
        shading="auto",
        cmap="cividis",
        rasterized=True,
        edgecolors="none",
        linewidth=0.0,
        vmin=float(np.min(correlated_gain)),
        vmax=float(np.max(correlated_gain)),
    )
    contours_d = d.contour(
        phase_grid,
        rho_grid,
        correlated_gain,
        levels=[1.94, 1.95, 1.96],
        colors="white",
        linewidths=0.7,
    )
    d.clabel(contours_d, fmt="%.2f", colors="white", fontsize=7, inline=True)
    d.set(
        xlabel=r"Intrinsic correlation phase $\phi_C$ (rad)",
        ylabel=r"Correlation magnitude $\rho$",
    )
    fig.colorbar(
        image_map, ax=d, pad=0.01, fraction=0.045, shrink=0.96, aspect=28, label="Gain at 10 Hz"
    )
    label_panel(d, "(d)")
    save_figure(fig, "fig7_validity_envelope")

    summary: dict[str, float] = {"optical_fraction_10hz": optical_fraction}
    for tag, epsilon_ba in [("e01", 0.01), ("e10", 0.10), ("e30", 0.30)]:
        optimum = stressed_optimum_r(epsilon_ba, optical_fraction)
        summary[f"backaction_optimum_db_{tag}"] = float(
            10.0 * np.log10(np.exp(2.0 * optimum))
        )
        summary[f"backaction_optimized_gain_{tag}"] = float(
            stressed_gain(optimum, epsilon_ba, optical_fraction)
        )
    for frequency in (10.0, 300.0):
        local = model_noise(np.asarray([frequency]), 1.0)
        local_fraction = float(
            local["shot"][0] ** 2 / local["total_coherent"][0] ** 2
        )
        optimum = stressed_optimum_r(0.10, local_fraction)
        phases = np.linspace(-np.pi, np.pi, 720)
        gains = []
        for phase in phases:
            numerator = correlated_total_psd(
                frequency, 0.0, CORRELATION_STRESS_MAX, float(phase)
            )
            denominator = correlated_total_psd(
                frequency,
                optimum,
                CORRELATION_STRESS_MAX,
                float(phase),
                epsilon_ba=0.10,
            )
            gains.append(math.sqrt(numerator / denominator))
        summary[f"correlated_gain_min_{int(frequency)}hz"] = float(np.min(gains))
        summary[f"correlated_gain_max_{int(frequency)}hz"] = float(np.max(gains))
    return summary


def make_fig8(
    freq: np.ndarray,
    noise: dict[str, np.ndarray],
    write_calibration_data: bool = True,
) -> dict[str, float]:
    fig, ax = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    a, b, c, d = ax.flat
    times = np.logspace(0, 7, 220)
    reference_coh = model_noise(np.asarray([10.0]), 1.0)
    reference_sq = model_noise(
        np.asarray([10.0]), effective_variance(R_FID, ETA_FID, PHASE_FID)
    )
    coh = float(reference_coh["total_coherent"][0])
    sq = float(reference_sq["total_squeezed"][0])
    tau10 = float(coherence_time(np.asarray([10.0]))[0])
    line_factor = np.asarray(
        [
            stochastic_line_factor(np.asarray([10.0]), float(obs_time))[0]
            for obs_time in times
        ]
    )
    coherent_coefficients = np.asarray(
        [
            global_gaussian_target(float(obs_time)) / math.sqrt(2.0)
            for obs_time in times
        ]
    )
    a.loglog(
        times,
        coherent_coefficients * coh / np.sqrt(times) * line_factor,
        color=BLUE,
        lw=1.2,
        label="coherent probe",
    )
    a.loglog(
        times,
        coherent_coefficients * sq / np.sqrt(times) * line_factor,
        color=ORANGE,
        lw=1.2,
        label="squeezed probe",
    )
    a.loglog(
        times,
        coherent_coefficients * coh / np.sqrt(times),
        color=GRAY,
        ls="--",
        lw=0.8,
        label="deterministic rms tone",
    )
    a.axvline(
        tau10, color=GRAY, ls=":", lw=0.7, label=r"$\tau_{\rm DM}(10\,\mathrm{Hz})$"
    )
    a.set(xlabel="Observation time (s)", ylabel="Asimov target field")
    a.legend(ncol=2, loc="upper right", fontsize=7.2, columnspacing=0.5, handlelength=1.2)
    label_panel(a, "(a)")

    calibration = finite_sample_calibration_curve()
    b.semilogx(
        calibration["time_bandwidth"],
        calibration["saddle_ratio"],
        color=BLUE,
        label="finite-sample saddlepoint",
    )
    b.errorbar(
        calibration["validation_bandwidth"],
        calibration["validation_ratio"],
        yerr=calibration["validation_error"],
        fmt="o",
        ms=3.5,
        capsize=2,
        elinewidth=0.8,
        capthick=0.8,
        color=ORANGE,
        ecolor=ORANGE,
        label="simulation-validated locations",
    )
    b.axhline(1.0, color=GRAY, ls=":", lw=0.7)
    b.set(
        xlabel=r"Line resolution $T\Delta f_a$",
        ylabel=r"Finite-sample / Asimov field",
    )
    b.legend(loc="lower left", fontsize=7.4, handlelength=1.4)
    label_panel(b, "(b)")

    # Weak-signal Fisher information for the covariance-power parameter P =
    # B_rms^2 scales as the inverse squared noise PSD.
    fisher_coh = 1.0 / noise["total_coherent"] ** 4
    fisher_sq = 1.0 / noise["total_squeezed"] ** 4
    c.loglog(freq, fisher_sq / fisher_coh, color=GREEN, lw=1.2)
    c.set(xlabel="Signal frequency (Hz)", ylabel="Power Fisher-information ratio")
    label_panel(c, "(c)")

    d.errorbar(
        calibration["validation_bandwidth"],
        calibration["validation_tail_ratio"],
        yerr=calibration["validation_tail_error"],
        fmt="s",
        ms=3.5,
        capsize=2,
        elinewidth=0.8,
        capthick=0.8,
        color=BLUE,
        ecolor=BLUE,
        label="null tail / target",
    )
    d.errorbar(
        calibration["validation_bandwidth"],
        calibration["validation_detection_ratio"],
        yerr=calibration["validation_detection_error"],
        fmt="o",
        ms=3.5,
        capsize=2,
        elinewidth=0.8,
        capthick=0.8,
        color=ORANGE,
        ecolor=ORANGE,
        label="detection / 0.5",
    )
    d.axhline(1.0, color=BLACK, ls="--", lw=0.8)
    d.set(
        xlabel=r"Line resolution $T\Delta f_a$",
        ylabel="Monte Carlo calibration ratio",
        xscale="log",
    )
    d.legend(loc="best")
    label_panel(d, "(d)")
    if write_calibration_data:
        write_csv(
            DATA / "finite_sample_calibration.csv",
            ["T_Delta_f", "finite_sample_to_asimov_field_ratio"],
            [
                [float(xx), float(yy)]
                for xx, yy in zip(
                    calibration["time_bandwidth"], calibration["saddle_ratio"]
                )
            ],
        )
        write_csv(
            DATA / "finite_sample_tail_validation.csv",
            [
                "T_Delta_f",
                "finite_sample_to_asimov_field_ratio",
                "importance_tail_to_target_tail",
                "importance_tail_ratio_standard_error",
                "detection_probability_to_0p5",
                "detection_ratio_standard_error",
                "draws",
            ],
            [
                [
                    float(xx),
                    float(yy),
                    float(tt),
                    float(ee),
                    float(dd),
                    float(de),
                    CALIBRATION_MC_DRAWS,
                ]
                for xx, yy, tt, ee, dd, de in zip(
                    calibration["validation_bandwidth"],
                    calibration["validation_ratio"],
                    calibration["validation_tail_ratio"],
                    calibration["validation_tail_error"],
                    calibration["validation_detection_ratio"],
                    calibration["validation_detection_error"],
                )
            ],
        )
    save_figure(fig, "fig8_scaling")
    fisher_ten = (coh / sq) ** 4
    return {
        "fisher_ratio_10hz": float(fisher_ten),
        "fisher_ratio_max": float(np.max(fisher_sq / fisher_coh)),
        "finite_sample_ratio_unresolved": float(calibration["saddle_ratio"][0]),
        "finite_sample_ratio_resolved_10": float(
            finite_sample_field_correction(np.asarray([10.0]))[0]
        ),
        "importance_tail_max_relative_error": float(
            calibration["maximum_tail_relative_error"]
        ),
        "detection_probability_max_relative_error": float(
            calibration["maximum_detection_relative_error"]
        ),
    }


def write_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_macros(summary: dict[str, float]) -> None:
    def fmt(value: float, digits: int = 3) -> str:
        return f"{value:.{digits}f}"

    lines = [
        "% Generated by codes/squeezed_opm_projections.py. Do not edit by hand.",
        f"\\newcommand{{\\SeedValue}}{{{SEED}}}",
        f"\\newcommand{{\\FiducialVariance}}{{{fmt(summary['vfid'], 4)}}}",
        f"\\newcommand{{\\OptimumR}}{{{fmt(R_FID, 3)}}}",
        f"\\newcommand{{\\FiducialDb}}{{{fmt(summary['optimal_source_db_10hz'], 2)}}}",
        f"\\newcommand{{\\FiducialGain}}{{{fmt(summary['gain_10hz'], 3)}}}",
        f"\\newcommand{{\\GainThreeHundred}}{{{fmt(summary['gain_300hz'], 3)}}}",
        f"\\newcommand{{\\OptimumSqueezing}}{{{fmt(summary['optimal_source_db_10hz'], 2)}}}",
        f"\\newcommand{{\\OptimumGain}}{{{fmt(summary['optimal_gain_10hz'], 3)}}}",
        f"\\newcommand{{\\FieldCoherent}}{{{fmt(summary['field_target_coherent_10hz'], 5)}}}",
        f"\\newcommand{{\\FieldSqueezed}}{{{fmt(summary['field_target_squeezed_10hz'], 5)}}}",
        f"\\newcommand{{\\FieldCoherentDet}}{{{fmt(summary['field_target_coherent_deterministic_10hz'], 5)}}}",
        f"\\newcommand{{\\FieldSqueezedDet}}{{{fmt(summary['field_target_squeezed_deterministic_10hz'], 5)}}}",
        f"\\newcommand{{\\LocalQFive}}{{{fmt(LOCAL_Q5, 6)}}}",
        f"\\newcommand{{\\CoherentAmpCoefficient}}{{{fmt(math.sqrt(required_noncentrality()), 6)}}}",
        f"\\newcommand{{\\GlobalGaussianTarget}}{{{fmt(summary['global_gaussian_target'], 4)}}}",
        f"\\newcommand{{\\GlobalTrialCount}}{{{int(summary['global_trial_count'])}}}",
        f"\\newcommand{{\\ScanOversampling}}{{{int(summary['scan_oversampling'])}}}",
        f"\\newcommand{{\\UnresolvedCalibrationFactor}}{{{fmt(summary['unresolved_exact_to_asimov_amplitude_factor'], 3)}}}",
        f"\\newcommand{{\\TauDMTen}}{{{fmt(summary['coherence_time_10hz_s'], 1)}}}",
        f"\\newcommand{{\\TauDMThreeHundred}}{{{fmt(summary['coherence_time_300hz_s'], 2)}}}",
        f"\\newcommand{{\\LineBinsTen}}{{{int(summary['line_bins_10hz'])}}}",
        f"\\newcommand{{\\LineBinsThreeHundred}}{{{int(summary['line_bins_300hz'])}}}",
        f"\\newcommand{{\\StochasticFactorTen}}{{{fmt(summary['stochastic_line_factor_10hz'], 3)}}}",
        f"\\newcommand{{\\StochasticFactorThreeHundred}}{{{fmt(summary['stochastic_line_factor_300hz'], 3)}}}",
        "\\newcommand{\\AxionCouplingCoherentTen}{\\ensuremath{"
        + f"{summary['axion_coupling_coherent_10hz']/1.0e-10:.3f}"
        + "\\times10^{-10}}}",
        "\\newcommand{\\AxionCouplingSqueezedTen}{\\ensuremath{"
        + f"{summary['axion_coupling_squeezed_10hz']/1.0e-10:.3f}"
        + "\\times10^{-10}}}",
        f"\\newcommand{{\\MCMedian}}{{{fmt(summary['mc_median_gain'], 3)}}}",
        f"\\newcommand{{\\MCPfive}}{{{fmt(summary['mc_p05_gain'], 3)}}}",
        f"\\newcommand{{\\MCPninetyfive}}{{{fmt(summary['mc_p95_gain'], 3)}}}",
        f"\\newcommand{{\\MCAltMedian}}{{{fmt(summary['mc_alt_median_gain'], 3)}}}",
        f"\\newcommand{{\\MCAltPfive}}{{{fmt(summary['mc_alt_p05_gain'], 3)}}}",
        f"\\newcommand{{\\MCAltPninetyfive}}{{{fmt(summary['mc_alt_p95_gain'], 3)}}}",
        f"\\newcommand{{\\MCAboveTen}}{{{100.0*summary['mc_fraction_above_1p1']:.1f}}}",
        f"\\newcommand{{\\SNRCoherent}}{{{fmt(summary['coherent_snr_1ks'], 2)}}}",
        f"\\newcommand{{\\SNRSqueezed}}{{{fmt(summary['squeezed_snr_1ks'], 2)}}}",
        f"\\newcommand{{\\PeriodogramRatio}}{{{fmt(summary['median_periodogram_ratio'], 3)}}}",
        f"\\newcommand{{\\OpticalFractionTen}}{{{fmt(summary['optical_fraction_10hz'], 3)}}}",
        f"\\newcommand{{\\BackactionOptDbOne}}{{{fmt(summary['backaction_optimum_db_e01'], 2)}}}",
        f"\\newcommand{{\\BackactionOptDbTen}}{{{fmt(summary['backaction_optimum_db_e10'], 2)}}}",
        f"\\newcommand{{\\BackactionOptDbThirty}}{{{fmt(summary['backaction_optimum_db_e30'], 2)}}}",
        f"\\newcommand{{\\BackactionGainOne}}{{{fmt(summary['backaction_optimized_gain_e01'], 3)}}}",
        f"\\newcommand{{\\BackactionGainTen}}{{{fmt(summary['backaction_optimized_gain_e10'], 3)}}}",
        f"\\newcommand{{\\BackactionGainThirty}}{{{fmt(summary['backaction_optimized_gain_e30'], 3)}}}",
        f"\\newcommand{{\\CorrelatedGainTenMin}}{{{fmt(summary['correlated_gain_min_10hz'], 3)}}}",
        f"\\newcommand{{\\CorrelatedGainTenMax}}{{{fmt(summary['correlated_gain_max_10hz'], 3)}}}",
        f"\\newcommand{{\\CorrelatedGainThreeHundredMin}}{{{fmt(summary['correlated_gain_min_300hz'], 3)}}}",
        f"\\newcommand{{\\CorrelatedGainThreeHundredMax}}{{{fmt(summary['correlated_gain_max_300hz'], 3)}}}",
        f"\\newcommand{{\\FiniteSampleResolvedTen}}{{{fmt(summary['finite_sample_ratio_resolved_10'], 3)}}}",
        f"\\newcommand{{\\ImportanceTailError}}{{{100.0*summary['importance_tail_max_relative_error']:.2f}}}",
        f"\\newcommand{{\\DetectionCalibrationError}}{{{100.0*summary['detection_probability_max_relative_error']:.2f}}}",
        f"\\newcommand{{\\FisherTen}}{{{fmt(summary['fisher_ratio_10hz'], 3)}}}",
        f"\\newcommand{{\\FisherMax}}{{{fmt(summary['fisher_ratio_max'], 3)}}}",
        f"\\newcommand{{\\FrequencyLine}}{{{fmt(summary['f0'], 1)}}}",
    ]
    (BUILD / "results_macros.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_grid_and_noise(
    freq: np.ndarray, noise: dict[str, np.ndarray]
) -> dict[str, bool | int]:
    """Fail loudly on malformed synthetic inputs before any result is saved."""
    expected_keys = (
        "shot_out",
        "atom_in",
        "laser_out",
        "environment_in",
        "light_shift_in",
        "shot",
        "atom",
        "laser",
        "environment",
        "light_shift",
        "total_coherent",
        "total_squeezed",
    )
    grid_valid = bool(
        freq.ndim == 1
        and freq.size == 620
        and np.all(np.isfinite(freq))
        and np.all(freq > 0.0)
        and np.all(np.diff(freq) > 0.0)
        and np.unique(freq).size == freq.size
    )
    if not grid_valid:
        raise ValueError(
            "Frequency grid must contain 620 finite, positive, strictly increasing, unique samples"
        )
    dimensions_valid = True
    for key in expected_keys:
        if key not in noise or noise[key].shape != freq.shape:
            dimensions_valid = False
            break
        if not np.all(np.isfinite(noise[key])):
            raise ValueError(
                f"Noise component {key!r} contains missing or nonfinite values"
            )
    if not dimensions_valid:
        raise ValueError(
            "Noise components have missing keys or dimensions inconsistent with the frequency grid"
        )
    return {
        "frequency_grid_valid": grid_valid,
        "noise_dimensions_valid": dimensions_valid,
        "unique_frequency_samples": int(np.unique(freq).size),
    }


def validate_covariance_optimum() -> bool:
    """Check the analytic minimum against the implemented covariance model."""
    c2 = 0.5 * (1.0 + math.exp(-2.0 * PHASE_FID**2))
    r_opt = optimum_squeezing_r(PHASE_FID)
    v_opt = effective_variance(r_opt, ETA_FID, PHASE_FID)
    v_closed = 1.0 - ETA_FID + 2.0 * ETA_FID * math.sqrt(c2 * (1.0 - c2))
    delta = 1.0e-4
    v_left = effective_variance(r_opt - delta, ETA_FID, PHASE_FID)
    v_right = effective_variance(r_opt + delta, ETA_FID, PHASE_FID)
    checked = bool(
        np.isfinite(r_opt)
        and np.isfinite(v_opt)
        and abs(v_opt - v_closed) < 1.0e-12
        and v_opt <= v_left
        and v_opt <= v_right
    )
    if not checked:
        raise RuntimeError(
            "Analytic covariance optimum failed the closed-form or local-minimum check"
        )
    return checked


def validate_backaction_and_correlations() -> bool:
    """Check the phenomenological stress model and covariance bounds."""
    reference = model_noise(np.asarray([10.0]), 1.0)
    optical_fraction = float(
        reference["shot"][0] ** 2 / reference["total_coherent"][0] ** 2
    )
    if abs(stressed_optimum_r(0.0, optical_fraction) - R_FID) > 1.0e-12:
        raise RuntimeError("Zero-backaction optimum does not recover the covariance optimum")
    if abs(normalized_backaction_psd(0.0, 0.30, optical_fraction)) > 1.0e-15:
        raise RuntimeError("Excess backaction does not vanish for a coherent probe")
    reference_psd = (
        optical_fraction * effective_variance(R_FID, ETA_FID, PHASE_FID)
        + 1.0
        - optical_fraction
    )
    recovered = normalized_backaction_psd(R_FID, 0.10, optical_fraction) / reference_psd
    if abs(recovered - 0.10) > 1.0e-12:
        raise RuntimeError("Backaction fraction normalization failed")
    r_grid = np.linspace(0.0, 3.2, 5001)
    for epsilon_ba in (0.0, 0.01, 0.10, 0.30, 1.0):
        analytic = stressed_optimum_r(epsilon_ba, optical_fraction)
        gains = np.asarray(
            [stressed_gain(float(rr), epsilon_ba, optical_fraction) for rr in r_grid]
        )
        numerical = float(r_grid[int(np.argmax(gains))])
        if abs(analytic - numerical) > 1.1 * float(r_grid[1] - r_grid[0]):
            raise RuntimeError("Analytic and numerical backaction optima disagree")
    for frequency in (10.0, 300.0):
        for rho in (0.0, 0.25, CORRELATION_STRESS_MAX):
            determinant_factor = 1.0 - rho**2
            if determinant_factor < 0.0:
                raise RuntimeError("Correlation covariance is not positive semidefinite")
            for phase in np.linspace(-np.pi, np.pi, 25):
                for rr in (0.0, 1.0, R_FID):
                    value = correlated_total_psd(
                        frequency,
                        rr,
                        rho,
                        float(phase),
                        epsilon_ba=0.10,
                    )
                    if not np.isfinite(value) or value <= 0.0:
                        raise RuntimeError("Correlation stress produced a nonpositive PSD")
    integer_probe = model_noise(np.asarray([10], dtype=int), 1.0)
    if abs(float(integer_probe["atom"][0]) - NOISE_LEVELS["atom_in"]) > 1.0e-15:
        raise RuntimeError("Integer-frequency calibration silently changed a noise amplitude")
    return True


def validate_search_equations() -> bool:
    """Verify the deterministic calibration and the stochastic Asimov equations."""
    dof = 2
    critical = float(chi2.isf(LOCAL_FALSE_ALARM, dof))
    false_alarm = float(chi2.sf(critical, dof))
    noncentrality = required_noncentrality()
    detection = float(ncx2.sf(critical, dof, noncentrality))
    if (
        not np.isfinite(critical)
        or not np.isfinite(noncentrality)
        or abs(false_alarm - LOCAL_FALSE_ALARM) > 1.0e-12 * LOCAL_FALSE_ALARM
        or abs(detection - DETECTION_PROBABILITY) > 1.0e-10
    ):
        raise RuntimeError(
            f"Search calibration failed: "
            f"false_alarm={false_alarm}, detection={detection}"
        )
    for frequency in (0.01, 10.0, 300.0, 1.0e4):
        weights = maxwellian_gradient_bin_weights(frequency, T_PROJECTION)
        power_ratio = required_stochastic_power_ratio(frequency, T_PROJECTION)
        target = global_gaussian_target(T_PROJECTION) ** 2
        time_bandwidth = T_PROJECTION * frequency * V0_DM**2 / 2.0
        if time_bandwidth <= 11.0:
            modes = finite_window_eigenfractions(time_bandwidth)
            calibrated_q = whittle_asimov_q(power_ratio, modes)
        elif time_bandwidth >= 13.0:
            calibrated_q = whittle_asimov_q(power_ratio, weights)
        else:
            finite_power = required_power_ratio_from_modes(
                finite_window_eigenfractions(time_bandwidth), target
            )
            whittle_power = required_power_ratio_from_modes(weights, target)
            relative_overlap = abs(math.sqrt(finite_power / whittle_power) - 1.0)
            if relative_overlap > 0.005:
                raise RuntimeError(
                    "Finite-window and Whittle thresholds disagree in the overlap"
                )
            calibrated_q = target
        if (
            abs(float(np.sum(weights)) - 1.0) > 1.0e-12
            or abs(calibrated_q - target) > 1.0e-8
        ):
            raise RuntimeError(f"Stochastic ULDM Asimov equation failed at {frequency} Hz")

    for time_bandwidth in (11.0, 12.0, 13.0):
        test_frequency = 10.0
        test_time = 2.0 * time_bandwidth / (test_frequency * V0_DM**2)
        finite_power = required_power_ratio_from_modes(
            finite_window_eigenfractions(time_bandwidth),
            global_gaussian_target(test_time) ** 2,
        )
        whittle_power = required_power_ratio_from_modes(
            maxwellian_gradient_bin_weights(test_frequency, test_time),
            global_gaussian_target(test_time) ** 2,
        )
        if abs(math.sqrt(finite_power / whittle_power) - 1.0) > 0.005:
            raise RuntimeError(
                "Finite-window and Whittle thresholds disagree in the overlap"
            )

    # More observing time must not worsen the optimized Asimov field threshold.
    test_times = np.logspace(2.0, 5.0, 40)
    for frequency in (0.01, 10.0, 300.0, 1.0e4):
        coefficients = np.asarray(
            [
                stochastic_line_coefficient(np.asarray([frequency]), float(tt))[0]
                for tt in test_times
            ]
        )
        amplitudes = coefficients / np.sqrt(test_times)
        if np.any(
            np.diff(amplitudes) > 1.0e-11 * np.maximum(amplitudes[:-1], 1.0)
        ):
            raise RuntimeError(
                f"ULDM threshold is not monotone with time at {frequency} Hz"
            )

    fields = np.asarray([1.0e-6, 1.0e-2, 10.0])
    round_trip = electron_equivalent_field_ft(axion_electron_coupling(fields))
    if not np.allclose(round_trip, fields, rtol=1.0e-12, atol=0.0):
        raise RuntimeError("Axion coupling conversion failed its round-trip check")

    # Exact single-mode check against the analytic exponential-power threshold.
    false_alarm = global_false_alarm_probability(T_PROJECTION)
    critical_power = -math.log(false_alarm)
    exact_power_ratio = critical_power / (-math.log(DETECTION_PROBABILITY)) - 1.0
    exact_detection = math.exp(-critical_power / (1.0 + exact_power_ratio))
    factor = unresolved_exact_to_asimov_amplitude_factor(T_PROJECTION)
    if abs(exact_detection - DETECTION_PROBABILITY) > 1.0e-12 or not 1.15 < factor < 1.21:
        raise RuntimeError("Unresolved-mode exact/Asimov comparison failed")
    # The plotted correction is anchored to the analytic unresolved limit and
    # stays finite through the matching interval.
    correction_probe = np.asarray(
        [
            finite_sample_ratio_for_bandwidth(float(u), false_alarm, T_PROJECTION)
            for u in (1.0e-4, 0.01, 0.10, 11.0, 12.0, 13.0, 100.0)
        ]
    )
    if (
        not np.all(np.isfinite(correction_probe))
        or abs(correction_probe[0] - factor) > 1.0e-12
        or np.any(correction_probe < 1.0)
        or np.max(np.abs(np.diff(correction_probe[3:6]))) > 0.01
    ):
        raise RuntimeError("Finite-mode calibration anchor or overlap check failed")
    return True


def validate_complex_response() -> bool:
    """Check the sign and magnitude implied by the Fourier convention."""
    test_freq = np.asarray([0.0, 10.0, 300.0])
    response = normalized_transfer(test_freq)
    expected_magnitude = 1.0 / np.sqrt(1.0 + (test_freq / 135.0) ** 2)
    checked = bool(
        abs(response[0] - 1.0) < 1.0e-15
        and np.allclose(np.abs(response), expected_magnitude, rtol=0.0, atol=1.0e-14)
        and np.all(np.imag(response[1:]) < 0.0)
        and np.all(np.angle(response[1:]) < 0.0)
    )
    if not checked:
        raise RuntimeError("Complex OPM response failed the Fourier-sign check")
    return checked


def validate_line_noise_flatness(
    freq: np.ndarray, noise: dict[str, np.ndarray]
) -> bool:
    """Check the locally flat-noise approximation over each ULDM template."""
    test_frequencies = np.logspace(
        math.log10(SCAN_F_MIN), math.log10(SCAN_F_MAX), 80
    )
    for key in ("total_coherent", "total_squeezed"):
        log_noise = np.log(np.asarray(noise[key], dtype=float))
        for frequency in test_frequencies:
            scale = frequency * V0_DM**2 / 2.0
            upper_offset = float(
                gammaincinv(MAXWELLIAN_GRADIENT_SHAPE, 1.0 - LINE_TAIL_PROBABILITY)
            ) * scale
            upper_frequency = min(float(freq[-1]), frequency + upper_offset)
            center = float(
                np.exp(np.interp(np.log(frequency), np.log(freq), log_noise))
            )
            upper = float(
                np.exp(np.interp(np.log(upper_frequency), np.log(freq), log_noise))
            )
            if abs(upper / center - 1.0) > 0.005:
                raise RuntimeError(
                    f"Noise varies by more than 0.5% across the ULDM line at {frequency} Hz"
                )
    return True


def main(figures_only: bool = False) -> None:
    setup_style()
    rng = np.random.default_rng(SEED)
    freq = np.logspace(-2, 4, 620)
    rfid, eta_fid, phase_fid = R_FID, ETA_FID, PHASE_FID
    vfid = effective_variance(rfid, eta_fid, phase_fid)
    noise = model_noise(freq, vfid)
    validation = validate_grid_and_noise(freq, noise)
    covariance_optimum_checked = validate_covariance_optimum()
    backaction_correlation_checked = validate_backaction_and_correlations()
    search_equations_checked = validate_search_equations()
    nystrom_convergence_checked, nystrom_max_relative_shift = (
        validate_nystrom_convergence()
    )
    complex_response_checked = validate_complex_response()
    line_noise_flatness_checked = validate_line_noise_flatness(freq, noise)

    make_fig1(freq, noise)
    make_fig2(freq, noise)
    synth = make_fig3(rng)
    sens = make_fig4(freq, noise, vfid)
    opt = make_fig5(freq, noise)
    robust = make_fig6(rng, freq, noise)
    alternate_mc = monte_carlo_statistics(np.random.default_rng(SEED + 1))
    validity_stress = make_fig7(freq, noise)
    scaling = make_fig8(freq, noise, write_calibration_data=not figures_only)

    summary: dict[str, float] = {
        "vfid": vfid,
        "local_q5": LOCAL_Q5,
        "local_false_alarm_probability": LOCAL_FALSE_ALARM,
        "deterministic_validation_detection_probability": DETECTION_PROBABILITY,
        "coherent_amplitude_coefficient": math.sqrt(required_noncentrality()),
        "v0_dm": V0_DM,
        "global_trial_count": N_SCAN_TRIALS,
        "scan_oversampling": SCAN_OVERSAMPLING,
        "global_false_alarm_probability_per_bin": GLOBAL_FALSE_ALARM,
        "global_gaussian_target": GLOBAL_GAUSSIAN_Z,
        "unresolved_exact_to_asimov_amplitude_factor": (
            unresolved_exact_to_asimov_amplitude_factor(T_PROJECTION)
        ),
        "rho_dm_GeV_cm3": RHO_DM_GEV_CM3,
        "projection_time_s": T_PROJECTION,
        **synth,
        **sens,
        **opt,
        **robust,
        **validity_stress,
        **scaling,
    }
    summary["mc_alt_median_gain"] = float(np.median(alternate_mc["gains"]))
    summary["mc_alt_p05_gain"] = float(np.percentile(alternate_mc["gains"], 5))
    summary["mc_alt_p95_gain"] = float(np.percentile(alternate_mc["gains"], 95))
    summary["response_phase_10hz_rad"] = float(
        np.angle(normalized_transfer(np.asarray([10.0]))[0])
    )
    summary["response_phase_300hz_rad"] = float(
        np.angle(normalized_transfer(np.asarray([300.0]))[0])
    )
    if figures_only:
        reference_summary = json.loads(
            (DATA / "summary.json").read_text(encoding="utf-8")
        )
        differences = {
            key: (reference_summary[key], value)
            for key, value in summary.items()
            if key in reference_summary
            and not math.isclose(
                float(reference_summary[key]),
                float(value),
                rel_tol=2.0e-12,
                abs_tol=1.0e-14,
            )
        }
        if differences:
            raise RuntimeError(
                f"Figure regeneration changed numerical summaries: {differences}"
            )
        print(
            json.dumps(
                {
                    "figures": 8,
                    "data_outputs_written": False,
                    "unchanged_summary_values": len(
                        set(reference_summary).intersection(summary)
                    ),
                },
                indent=2,
            )
        )
        return

    write_macros(summary)

    write_csv(
        DATA / "noise_spectrum.csv",
        [
            "frequency_Hz",
            "shot_equiv_input_fT_sqrtHz",
            "atom_process_input_fT_sqrtHz",
            "laser_equiv_input_fT_sqrtHz",
            "environment_input_fT_sqrtHz",
            "light_shift_input_fT_sqrtHz",
            "total_coherent_fT_sqrtHz",
            "total_squeezed_fT_sqrtHz",
        ],
        [
            [
                float(freq[i]),
                float(noise["shot"][i]),
                float(noise["atom"][i]),
                float(noise["laser"][i]),
                float(noise["environment"][i]),
                float(noise["light_shift"][i]),
                float(noise["total_coherent"][i]),
                float(noise["total_squeezed"][i]),
            ]
            for i in range(freq.size)
        ],
    )
    threshold_coh_det = deterministic_asimov_target(noise["total_coherent"], T_PROJECTION)
    threshold_sq_det = deterministic_asimov_target(noise["total_squeezed"], T_PROJECTION)
    threshold_coh_uldm = ul_dm_asimov_target(noise["total_coherent"], freq, T_PROJECTION)
    threshold_sq_uldm = ul_dm_asimov_target(noise["total_squeezed"], freq, T_PROJECTION)
    finite_correction = finite_sample_field_correction(freq)
    write_csv(
        DATA / "sensitivity_projection.csv",
        [
            "frequency_Hz",
            "field_target_coherent_deterministic_asimov_global5sigma_fT",
            "field_target_squeezed_deterministic_asimov_global5sigma_fT",
            "gain_deterministic",
            "field_target_coherent_uldm_expectedLLR_global5sigma_fT",
            "field_target_squeezed_uldm_expectedLLR_global5sigma_fT",
            "finite_sample_to_asimov_field_correction",
            "field_target_coherent_uldm_50pct_power_global5sigma_fT",
            "field_target_squeezed_uldm_50pct_power_global5sigma_fT",
            "gain_uldm",
            "tau_DM_s",
            "maxwellian_template_bins",
            "stochastic_line_asimov_factor",
            "axion_mass_eV",
            "gae_coherent_benchmark",
            "gae_squeezed_benchmark",
        ],
        [
            [
                float(freq[i]),
                float(threshold_coh_det[i]),
                float(threshold_sq_det[i]),
                float(threshold_coh_det[i] / threshold_sq_det[i]),
                float(threshold_coh_uldm[i]),
                float(threshold_sq_uldm[i]),
                float(finite_correction[i]),
                float(threshold_coh_uldm[i] * finite_correction[i]),
                float(threshold_sq_uldm[i] * finite_correction[i]),
                float(threshold_coh_uldm[i] / threshold_sq_uldm[i]),
                float(coherence_time(np.asarray([freq[i]]))[0]),
                int(line_bin_count(np.asarray([freq[i]]), T_PROJECTION)[0]),
                float(stochastic_line_factor(np.asarray([freq[i]]), T_PROJECTION)[0]),
                float(axion_mass_ev(np.asarray([freq[i]]))[0]),
                float(
                    axion_electron_coupling(np.asarray([threshold_coh_uldm[i]]))[0]
                ),
                float(
                    axion_electron_coupling(np.asarray([threshold_sq_uldm[i]]))[0]
                ),
            ]
            for i in range(freq.size)
        ],
    )
    (DATA / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checks = {
        "finite_noise": bool(
            np.all(np.isfinite(noise["total_coherent"]))
            and np.all(np.isfinite(noise["total_squeezed"]))
        ),
        "squeezed_not_worse_fiducial": bool(
            np.all(noise["total_squeezed"] <= noise["total_coherent"])
        ),
        "covariance_optimum_checked": covariance_optimum_checked,
        "backaction_correlation_checked": backaction_correlation_checked,
        "search_equations_checked": search_equations_checked,
        "nystrom_convergence_checked": nystrom_convergence_checked,
        "nystrom_max_relative_field_shift": nystrom_max_relative_shift,
        "complex_response_checked": complex_response_checked,
        "line_noise_flatness_checked": line_noise_flatness_checked,
        "finite_sample_calibration_checked": bool(
            1.0 <= summary["finite_sample_ratio_resolved_10"] < 1.15
            and summary["importance_tail_max_relative_error"] < 0.05
            and summary["detection_probability_max_relative_error"] < 0.05
            and abs(
                summary["finite_sample_ratio_unresolved"]
                - summary["unresolved_exact_to_asimov_amplitude_factor"]
            )
            < 0.01
        ),
        "figures": len(list(FIG.glob("fig*.pdf"))),
        "rows": int(freq.size),
        "mc_seed_sensitivity_checked": bool(
            np.all(np.isfinite(alternate_mc["gains"]))
            and abs(summary["mc_alt_median_gain"] - summary["mc_median_gain"]) < 0.05
        ),
        **validation,
    }
    if (
        not checks["finite_noise"]
        or not checks["squeezed_not_worse_fiducial"]
        or checks["figures"] != 8
        or not checks["frequency_grid_valid"]
        or not checks["noise_dimensions_valid"]
        or not checks["covariance_optimum_checked"]
        or not checks["backaction_correlation_checked"]
        or not checks["search_equations_checked"]
        or not checks["nystrom_convergence_checked"]
        or not checks["complex_response_checked"]
        or not checks["line_noise_flatness_checked"]
        or not checks["finite_sample_calibration_checked"]
        or not checks["mc_seed_sensitivity_checked"]
    ):
        raise RuntimeError(f"Model checks failed: {checks}")
    (BUILD / "checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(figures_only="--figures-only" in sys.argv[1:])
