"""
Higher-order terms II-V of the population Fisher matrix
(Gair et al. 2022, arXiv:2205.07893 Eq. 21).

    Gamma_II  = +1/2 sum_k d^2 log det(FIM_k - H_k) / dlambda^2
    Gamma_III = -1/2 sum_k d^2 Tr[(FIM_k - H_k)^{-1} FIM_k] / dlambda^2 / p_det_k
    Gamma_IV  = -    sum_k d^2 [P_k^T (FIM_k - H_k)^{-1} d_det_k] / dlambda^2
    Gamma_V   = -1/2 sum_k d^2 [P_k^T (FIM_k - H_k)^{-1} P_k] / dlambda^2

These corrections matter only when per-event measurement errors are not
negligible compared with the width of the population distribution; the dominant
Gamma_I lives in ``pop_fisher_term_I`` and ``pop_fisher`` assembles the total.

Everything here is nested finite differences (lambda-space outside,
theta-space inside), so numerical noise is the usual failure mode -- see the
long note in ``_hessians_correction_terms_fd`` on the choice of ``h_lam``.
"""

import logging, numpy
from dataclasses import dataclass, field

from load import POPULATION_KEYS, _default_cosmology

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------------
# Detection score:  d ln p_det(theta_k) / d theta
# ---------------------------------------------------------------------------


def compute_det_score(events, snr, snr_threshold, sigma_rho=1.0, p_det_min=1e-3):
    """
    Per-event detection gradient  d_k = d ln p_det(theta_k) / d theta,
    for the erfc detection model and the dominant inspiral SNR scaling:

        rho_opt ∝ Mc_det^{5/6}(1+z)^{5/6} / d_L(z)

    With x = (rho_th - rho) / (sqrt(2) sigma_rho) and p_det = 0.5 erfc(x),

        d ln p_det / d rho = 2 / (sqrt(2 pi) sigma_rho * erfcx(x)),

    where erfcx(x) = exp(x^2) erfc(x) is O(1/x) for large x, so this is always
    finite.  Beware: phi(x)/p_det with the standard normal PDF is wrong here
    (x is not the standardised variable) and diverges for rho << rho_th.

    Parameters
    ----------
    events        : dict with keys 'mass_1_source', 'mass_2_source', 'redshift'
    snr           : (N,) optimal matched-filter SNR per event
    snr_threshold : float – detection threshold rho_th (default 12.0)
    sigma_rho     : float – SNR uncertainty width (default 1.0)
    p_det_min     : float – events with p_det < p_det_min are zeroed out.
                    These are events well below the threshold; they would not
                    be in the detected sample for this threshold choice and
                    their large-x detection score should not enter the sum.
                    Default 1e-3.

    Returns
    -------
    det_score : (N, 3)  columns
                [d/dmass_1_source, d/dmass_2_source, d/dredshift]
    p_det     : (N,) detection probability per event
    """
    from scipy.special import erfc, erfcx
    from astropy import units as u

    mass_1 = events["mass_1_source"]
    mass_2 = events["mass_2_source"]
    redshift = events["redshift"]
    total_mass = mass_1 + mass_2

    cosmo = _default_cosmology()
    lum_dist = cosmo.luminosity_distance(redshift).to(u.Mpc).value

    # d ln d_L / dz via centred finite differences
    h_z = numpy.maximum(1e-4 * numpy.abs(redshift), 1e-6)
    dL_plus = cosmo.luminosity_distance(redshift + h_z).to(u.Mpc).value
    dL_minus = cosmo.luminosity_distance(redshift - h_z).to(u.Mpc).value
    dlndL_dz = (dL_plus - dL_minus) / (2.0 * h_z * lum_dist)

    # d ln Mc / d m1, m2  for  Mc = (m1 m2)^{3/5} / (m1+m2)^{1/5}
    dlnMc_dm1 = 3.0 / (5.0 * mass_1) - 1.0 / (5.0 * total_mass)
    dlnMc_dm2 = 3.0 / (5.0 * mass_2) - 1.0 / (5.0 * total_mass)

    # d ln rho / d theta  (dominant inspiral: rho ∝ Mc_det^{5/6} / d_L)
    dlnrho_dm1 = (5.0 / 6.0) * dlnMc_dm1
    dlnrho_dm2 = (5.0 / 6.0) * dlnMc_dm2
    dlnrho_dz = (5.0 / 6.0) / (1.0 + redshift) - dlndL_dz

    # erfc model: p_det = 0.5 erfc(x),  x = (rho_th - rho) / (sqrt(2) sigma_rho)
    x_arg = (snr_threshold - snr) / (numpy.sqrt(2.0) * sigma_rho)
    p_det = 0.5 * erfc(x_arg)

    # d ln p_det / d rho = 2 / (sqrt(2 pi) sigma_rho erfcx(x))
    # erfcx is O(1/x) for large x so this is always finite.
    dlnpdet_drho = 2.0 / (numpy.sqrt(2.0 * numpy.pi) * sigma_rho * erfcx(x_arg))

    # Zero out events below detection threshold: p_det < p_det_min means
    # the event would not be in the detected sample for this rho_th.
    below_threshold = p_det < p_det_min
    if below_threshold.any():
        logging.debug(
            f"compute_det_score: {below_threshold.sum()} events have p_det < {p_det_min}; "
            "setting their det_score to zero."
        )
        dlnpdet_drho = dlnpdet_drho.copy()
        dlnpdet_drho[below_threshold] = 0.0

    # d ln p_det / d theta_a = (d ln p_det / d rho) * rho * (d ln rho / d theta_a)
    det_score = numpy.column_stack(
        [
            dlnpdet_drho * snr * dlnrho_dm1,
            dlnpdet_drho * snr * dlnrho_dm2,
            dlnpdet_drho * snr * dlnrho_dz,
        ]
    )
    return det_score, p_det


# ---------------------------------------------------------------------------
# Physical-space derivatives of the population model
# ---------------------------------------------------------------------------


def _H_theta(model, events, parameters, phys_keys, h_rel=1e-3, h_abs=1e-6):
    """
    Hessian H_k^{ab} = d^2 ln p(theta_k | lambda) / d theta^a d theta^b
    for each event, computed via centred finite differences in theta-space.

    Parameters
    ----------
    model      : PopModel (or wrapper)
    events     : dict of per-event arrays
    parameters : dict of hyperparameters at which to evaluate
    phys_keys  : list of physical param keys in events dict (e.g. ['redshift'])
    h_rel      : relative FD step in theta-space
    h_abs      : minimum absolute FD step

    Returns
    -------
    H : (N_events, n_phys, n_phys)
    """
    N_events = len(events[phys_keys[0]])
    n_phys = len(phys_keys)
    hessian = numpy.zeros((N_events, n_phys, n_phys))
    lp0 = model.log_prob(events, parameters)

    for a, key_a in enumerate(phys_keys):
        val_a = events[key_a]
        h_a = numpy.maximum(h_rel * numpy.abs(val_a), h_abs)

        ev_plus = dict(events)
        ev_plus[key_a] = val_a + h_a
        ev_minus = dict(events)
        ev_minus[key_a] = val_a - h_a
        # When perturbing a mass, q = m2/m1 is a derived quantity used by
        # mass population models; keep it consistent.
        if key_a in ('mass_1_source', 'mass_2_source') and 'mass_ratio' in events:
            ev_plus['mass_ratio'] = ev_plus['mass_2_source'] / ev_plus['mass_1_source']
            ev_minus['mass_ratio'] = ev_minus['mass_2_source'] / ev_minus['mass_1_source']
        lp_plus = model.log_prob(ev_plus, parameters)
        lp_minus = model.log_prob(ev_minus, parameters)
        hessian[:, a, a] = (lp_plus + lp_minus - 2.0 * lp0) / h_a**2

        for b, key_b in enumerate(phys_keys[a + 1 :], start=a + 1):
            val_b = events[key_b]
            h_b = numpy.maximum(h_rel * numpy.abs(val_b), h_abs)

            ev_pp = dict(events)
            ev_pp[key_a] = val_a + h_a
            ev_pp[key_b] = val_b + h_b
            ev_pm = dict(events)
            ev_pm[key_a] = val_a + h_a
            ev_pm[key_b] = val_b - h_b
            ev_mp = dict(events)
            ev_mp[key_a] = val_a - h_a
            ev_mp[key_b] = val_b + h_b
            ev_mm = dict(events)
            ev_mm[key_a] = val_a - h_a
            ev_mm[key_b] = val_b - h_b
            if (
                key_a in ('mass_1_source', 'mass_2_source')
                or key_b in ('mass_1_source', 'mass_2_source')
            ) and 'mass_ratio' in events:
                for _ev in (ev_pp, ev_pm, ev_mp, ev_mm):
                    _ev['mass_ratio'] = _ev['mass_2_source'] / _ev['mass_1_source']

            hessian[:, a, b] = hessian[:, b, a] = (
                model.log_prob(ev_pp, parameters)
                - model.log_prob(ev_pm, parameters)
                - model.log_prob(ev_mp, parameters)
                + model.log_prob(ev_mm, parameters)
            ) / (4.0 * h_a * h_b)

    return hessian


def _P_theta(model, events, parameters, phys_keys, h_rel=1e-3, h_abs=1e-6):
    """
    Gradient P_k^a = d ln p(theta_k | lambda) / d theta^a for each event.

    Returns
    -------
    gradient : (N_events, n_phys)
    """
    N_events = len(events[phys_keys[0]])
    n_phys = len(phys_keys)
    gradient = numpy.zeros((N_events, n_phys))

    for a, key_a in enumerate(phys_keys):
        val_a = events[key_a]
        h_a = numpy.maximum(h_rel * numpy.abs(val_a), h_abs)
        ev_plus = dict(events)
        ev_plus[key_a] = val_a + h_a
        ev_minus = dict(events)
        ev_minus[key_a] = val_a - h_a
        if key_a in ('mass_1_source', 'mass_2_source') and 'mass_ratio' in events:
            ev_plus['mass_ratio'] = ev_plus['mass_2_source'] / ev_plus['mass_1_source']
            ev_minus['mass_ratio'] = ev_minus['mass_2_source'] / ev_minus['mass_1_source']
        gradient[:, a] = (
            model.log_prob(ev_plus, parameters) - model.log_prob(ev_minus, parameters)
        ) / (2.0 * h_a)

    return gradient


# ---------------------------------------------------------------------------
# Scalar functions per event (building blocks for terms II-V)
# ---------------------------------------------------------------------------


def _regularised_fisher(fim_sub, reg=1e-10):
    """
    F_eff = FIM + reg * max(diag(FIM)) * I, the regularised per-event Fisher.

    Term II is evaluated as log det(A) - log det(F_eff), so both halves must use
    the *same* regularised FIM or the identity no longer holds exactly.

    Returns F_eff (N, n, n).
    """
    diag_fim = numpy.diagonal(fim_sub, axis1=1, axis2=2)
    scale = numpy.maximum(diag_fim.max(axis=-1), 1.0) * reg
    eye_block = numpy.eye(fim_sub.shape[-1])[numpy.newaxis, :, :]
    return fim_sub + scale[:, numpy.newaxis, numpy.newaxis] * eye_block


def _correction_scalars(fim_eff, hessian, gradient, detection_score=None):
    """
    Evaluate the four scalar building blocks of terms II-V for each event.

        s_logdet[k] = log det(I - F_eff_k^{-1} H_k)
        s_trace[k]  = Tr[ A_k^{-1} H_k ]
        s_termV[k]  = P_k^T A_k^{-1} P_k
        s_termIV[k] = P_k^T A_k^{-1} d_det_k  (None if detection_score not given)

    where  A_k = F_eff_k - H_k.

    The first two are re-centred (log det(F_eff) and n_phys subtracted, which
    vanish under d^2/dlambda^2) because Eq. 21's raw log det(A) and
    Tr[A^{-1} FIM] are dominated by a large lambda-independent piece when
    FIM >> H; differencing the O(H/FIM) remainder instead is what lets all four
    terms share one FD path.  The residual reg * Tr[A^{-1}] is O(1e-10) and
    dropped.

    Parameters
    ----------
    fim_eff         : (N, n, n)  – regularised FIM, from ``_regularised_fisher``
    hessian         : (N, n, n)  – H_theta = d^2 ln p / d theta^2
    gradient        : (N, n)     – P_theta = d ln p / d theta
    detection_score : (N, n) or None

    Returns
    -------
    s_logdet, s_trace, s_termV, s_termIV  – each (N,); s_termIV may be None.
    """
    A_matrix = fim_eff - hessian
    N_events, n_phys, _ = A_matrix.shape

    # I - F_eff^{-1} H = F_eff^{-1} A, so its log-det is log det A - log det F_eff.
    eye_block = numpy.eye(n_phys)[numpy.newaxis, :, :]
    ratio = eye_block - numpy.linalg.solve(fim_eff, hessian)
    sign, s_logdet = numpy.linalg.slogdet(ratio)
    # Replace invalid log-dets with NaN (A singular or not positive definite,
    # i.e. the population is more informative than the data for this event).
    s_logdet = numpy.where(sign > 0, s_logdet, numpy.nan)

    # Inverse of A
    A_inv = numpy.zeros_like(A_matrix)
    for k in range(N_events):
        try:
            A_inv[k] = numpy.linalg.inv(A_matrix[k])
        except numpy.linalg.LinAlgError:
            A_inv[k] = numpy.linalg.pinv(A_matrix[k])

    # Tr[A^{-1} H]  (= Tr[A^{-1} FIM] up to a lambda-independent constant)
    s_trace = numpy.einsum("nij,nji->n", A_inv, hessian)

    # P^T A^{-1} P
    A_inv_P = numpy.einsum("nij,nj->ni", A_inv, gradient)
    s_termV = numpy.einsum("ni,ni->n", gradient, A_inv_P)

    s_termIV = None
    if detection_score is not None:
        A_inv_d = numpy.einsum("nij,nj->ni", A_inv, detection_score)
        s_termIV = numpy.einsum("ni,ni->n", gradient, A_inv_d)

    return s_logdet, s_trace, s_termV, s_termIV


# ---------------------------------------------------------------------------
# Lambda-Hessians of terms II-V: centred FD on the re-centred scalars
# ---------------------------------------------------------------------------


def _hessians_correction_terms_fd(
    model,
    events,
    fim_sub,
    free_names,
    params_vec,
    phys_keys,
    h_lam=3e-2,
    h_theta=1e-3,
    detection_score=None,
):
    """
    Compute the (N_free, N_free, N_events) second-derivative tensors for all
    four correction-term building blocks of Gair+2022 Eq. 21, by centred finite
    differences in lambda (3-point diagonal, 4-point off-diagonal), with
    H_theta and P_theta rebuilt by theta-FD at each lambda sample.  Re-centring
    the scalars (see ``_correction_scalars``) is what makes one FD path work
    for all four terms.

    **h_lam = 3e-2, not 1e-4, on purpose**: the theta-FD roundoff floor in
    H_theta propagates into the O(H/FIM) scalars and gets divided by h_lam^2,
    so at 1e-4 the result is pure noise (350-2600x too large vs gwfast, scaling
    as 1/h_lam^2).  The values plateau for h_lam in [1e-2, 1e-1], agreeing with
    gwfast to a few percent.  To re-check on a new model: recompute a diagonal
    at h_lam and 3*h_lam — entries that move like 1/h_lam^2 are noise.

    Parameters
    ----------
    model           : PopModel (or wrapper)
    events          : dict of per-event arrays
    fim_sub         : (N, n_phys, n_phys) per-event Fisher in physical basis
    free_names      : list of free hyperparameter names
    params_vec      : 1D array of free hyperparameter values (fiducial)
    phys_keys       : list of physical param keys
    h_lam           : relative FD step in lambda-space (default 3e-2, see above)
    h_theta         : relative FD step in theta-space  (default 1e-3)
    detection_score : (N, n_phys) detection gradient or None

    Returns
    -------
    d2_logdet : (N_free, N_free, N)   [Term II building block]
    d2_trace  : (N_free, N_free, N)   [Term III building block]
    d2_termV  : (N_free, N_free, N)   [Term V building block]
    d2_termIV : (N_free, N_free, N) or None  [Term IV building block]
    """
    N_lam = len(params_vec)
    N_events = len(events[phys_keys[0]])
    steps = numpy.array([max(h_lam * abs(params_vec[i]), 1e-8) for i in range(N_lam)])

    # FIM does not depend on lambda, so regularise it once.
    fim_eff = _regularised_fisher(fim_sub)

    def _make_params(vec):
        return {name: float(x) for name, x in zip(free_names, vec)}

    def _eval_scalars(lam_vec):
        parameters = _make_params(lam_vec)
        hessian = _H_theta(model, events, parameters, phys_keys, h_theta)
        gradient = _P_theta(model, events, parameters, phys_keys, h_theta)
        return _correction_scalars(fim_eff, hessian, gradient, detection_score)

    # Centre
    s0 = _eval_scalars(params_vec)

    n_not_pd = int(numpy.isnan(s0[0]).sum())
    if n_not_pd > 0:
        logging.warning(
            f"{n_not_pd} / {N_events} events have a non-positive-definite "
            f"(FIM - H_theta); their terms II-V contributions are dropped."
        )

    # Single-step cache
    cache_plus = {}
    cache_minus = {}
    for i in range(N_lam):
        lv_plus = params_vec.copy()
        lv_plus[i] += steps[i]
        lv_minus = params_vec.copy()
        lv_minus[i] -= steps[i]
        cache_plus[i] = _eval_scalars(lv_plus)
        cache_minus[i] = _eval_scalars(lv_minus)

    n_scalars = len(s0)
    d2_all = [
        numpy.zeros((N_lam, N_lam, N_events)) if s0[t] is not None else None
        for t in range(n_scalars)
    ]

    for i in range(N_lam):
        # Diagonal
        for t in range(n_scalars):
            if d2_all[t] is None:
                continue
            d2_all[t][i, i] = (
                cache_plus[i][t] + cache_minus[i][t] - 2.0 * s0[t]
            ) / steps[i] ** 2
        # Off-diagonal
        for j in range(i + 1, N_lam):
            lv_pp = params_vec.copy()
            lv_pp[i] += steps[i]
            lv_pp[j] += steps[j]
            lv_pm = params_vec.copy()
            lv_pm[i] += steps[i]
            lv_pm[j] -= steps[j]
            lv_mp = params_vec.copy()
            lv_mp[i] -= steps[i]
            lv_mp[j] += steps[j]
            lv_mm = params_vec.copy()
            lv_mm[i] -= steps[i]
            lv_mm[j] -= steps[j]
            s_pp = _eval_scalars(lv_pp)
            s_pm = _eval_scalars(lv_pm)
            s_mp = _eval_scalars(lv_mp)
            s_mm = _eval_scalars(lv_mm)
            for t in range(n_scalars):
                if d2_all[t] is None:
                    continue
                d2_all[t][i, j] = d2_all[t][j, i] = (
                    s_pp[t] - s_pm[t] - s_mp[t] + s_mm[t]
                ) / (4.0 * steps[i] * steps[j])

    d2_logdet, d2_trace, d2_termV, d2_termIV = d2_all
    return d2_logdet, d2_trace, d2_termV, d2_termIV


# ---------------------------------------------------------------------------
# Sub-block selection: marginalise over physical parameters the model ignores
# ---------------------------------------------------------------------------


def _marginalise_event_fisher(fisher_source_frame, phys_keys):
    """
    Reduce the (N, 3, 3) per-event Fisher to the parameters the model depends on.

    A naive slice would be wrong: dropping a row/column of a Fisher matrix fixes
    that parameter rather than marginalising over it.  The Schur complement

        F_AA - F_AB F_BB^{-1} F_BA

    is the correct marginal, so a redshift-only model gets a properly
    marginalised 1x1 event Fisher rather than the (redshift, redshift) entry.

    Parameters
    ----------
    fisher_source_frame : (N, 3, 3) ndarray
        Per-event Fisher in the ``load.POPULATION_KEYS`` basis, i.e.
        (mass_1_source, mass_2_source, redshift).
    phys_keys : list[str]
        Subset of ``load.POPULATION_KEYS`` the population model depends on.

    Returns
    -------
    (N, n_phys, n_phys) ndarray
    """
    number_of_events = len(fisher_source_frame)
    kept = [POPULATION_KEYS.index(key) for key in phys_keys]
    dropped = [index for index in range(len(POPULATION_KEYS)) if index not in kept]
    event_index = numpy.arange(number_of_events)

    if not dropped:
        return fisher_source_frame[numpy.ix_(event_index, kept, kept)]

    F_AA = fisher_source_frame[numpy.ix_(event_index, kept, kept)]
    F_AB = fisher_source_frame[numpy.ix_(event_index, kept, dropped)]
    F_BB = fisher_source_frame[numpy.ix_(event_index, dropped, dropped)]
    F_BA = fisher_source_frame[numpy.ix_(event_index, dropped, kept)]

    F_BB_inverse = numpy.zeros_like(F_BB)
    for k in range(number_of_events):
        try:
            F_BB_inverse[k] = numpy.linalg.inv(F_BB[k])
        except numpy.linalg.LinAlgError:
            F_BB_inverse[k] = numpy.linalg.pinv(F_BB[k])

    return F_AA - numpy.einsum("nab,nbc,ncd->nad", F_AB, F_BB_inverse, F_BA)


# ---------------------------------------------------------------------------
# Result container and driver for terms II-V
# ---------------------------------------------------------------------------


@dataclass
class CorrectionTerms:
    """
    The four sub-dominant Fisher contributions and their diagnostics.

    Attributes
    ----------
    fisher_II, fisher_III, fisher_IV, fisher_V : (n_free, n_free) ndarray
    p_det : ndarray or None
        Per-event detection probability actually used as the Term III weight.
        ``None`` means none was available and the complete-catalogue limit
        (p_det = 1) was assumed.
    detection_score : ndarray or None
        (N, n_phys) detection gradient used for Term IV; ``None`` means Term IV
        is identically zero.
    """

    fisher_II: numpy.ndarray
    fisher_III: numpy.ndarray
    fisher_IV: numpy.ndarray
    fisher_V: numpy.ndarray
    p_det: numpy.ndarray = None
    detection_score: numpy.ndarray = field(default=None, repr=False)

    @property
    def total(self):
        return self.fisher_II + self.fisher_III + self.fisher_IV + self.fisher_V


def compute_correction_terms(
    model,
    events,
    fisher_source_frame,
    free_names,
    params_vec,
    phys_keys,
    snr=None,
    snr_threshold=None,
    det_score=None,
    p_det_per_event=None,
    sigma_rho=1.0,
    p_det_min=1e-3,
    h_lam=3e-2,
    h_theta=1e-3,
):
    """
    Evaluate Gamma_II .. Gamma_V for a detected catalogue.

    Parameters
    ----------
    model : PopModel or _FreeParamWrapper
        Must expose ``log_prob(events, params)`` over exactly ``free_names``.
    events : dict
        Per-event source-frame arrays, already restricted to the events that
        enter the sum (i.e. those with finite Term I scores).
    fisher_source_frame : (N, 3, 3) ndarray
        Per-event Fisher in the ``load.POPULATION_KEYS`` basis, aligned with
        ``events``.  Marginalised down to ``phys_keys`` internally.
    free_names : list[str]
    params_vec : (n_free,) ndarray
        Fiducial values of the free hyperparameters, in ``free_names`` order.
    phys_keys : list[str]
        Physical parameters the model depends on.
    snr : (N,) ndarray, optional
        Per-event optimal SNR.  When given, both the Term IV detection gradient
        and the Term III weight are derived from it via the erfc model.
    snr_threshold : float, optional
        Required alongside ``snr``.
    det_score, p_det_per_event : ndarray, optional
        Pre-computed alternatives to deriving them from ``snr``; they take
        precedence when supplied.
    sigma_rho, p_det_min : float
        erfc detection-model width, and the floor below which an event's
        detection gradient is zeroed / its Term III weight is clipped.
    h_lam : float
        Relative FD step in lambda-space, default 3e-2.  Deliberately much
        larger than the Term I score step -- see
        ``_hessians_correction_terms_fd`` for why 1e-4 sits below the noise
        floor here.
    h_theta : float
        Relative FD step in theta-space, default 1e-3.

    Returns
    -------
    CorrectionTerms

    Notes
    -----
    Monte-Carlo weights: summing over a *detected* catalogue (drawn from
    p_det(theta) p(theta|lambda) / P_det(lambda)) means Terms I, II and V need
    no explicit weight, while Term III — weighted by p(theta|lambda) /
    P_det(lambda) in Eq. 21 — picks up a per-event 1/p_det(theta_k) from the
    change of measure.  Term IV's p_det(theta_k) in D_l cancels that factor
    exactly (``compute_det_score`` returns the log-derivative).  Term III
    substitutes D_kl -> FIM_kl, as gwfast's ``pop_function_hessian_termIII``
    also does.
    """
    number_of_events = len(events[phys_keys[0]])
    number_of_free = len(free_names)

    fim_sub = _marginalise_event_fisher(fisher_source_frame, phys_keys)

    # ------------------------------------------------------------------
    # Per-event detection probability (Term III weight) and detection
    # gradient (Term IV).  Explicit arguments take precedence over `snr`.
    # ------------------------------------------------------------------
    if snr is not None and len(snr) != number_of_events:
        raise ValueError(
            f"snr length ({len(snr)}) does not match the number of catalogue "
            f"events ({number_of_events}).  Pass the per-catalogue-event SNR "
            f"array (same snr_threshold cut as events / fisher_source_frame)."
        )

    if p_det_per_event is None and snr is not None:
        from scipy.special import erfc

        p_det_per_event = 0.5 * erfc(
            (snr_threshold - numpy.asarray(snr)) / (numpy.sqrt(2.0) * sigma_rho)
        )

    # Note on Term IV magnitude: d ln p_det / d rho is proportional to
    # 1/erfcx(x) with x = (snr_threshold - snr) / (sqrt(2)*sigma_rho).  For
    # events well above threshold this gradient is genuinely near zero, so
    # Term IV is physically small when the catalogue is highly complete.  It
    # grows when many events sit within ~sigma_rho of the threshold.
    if det_score is None and snr is not None:
        if not all(key in events for key in POPULATION_KEYS):
            logging.warning(
                "snr provided but events lacks "
                f"{[k for k in POPULATION_KEYS if k not in events]}; "
                "cannot compute the detection score.  Term IV will be zero."
            )
        else:
            logging.info(
                f"Computing det_score internally "
                f"(snr_threshold={snr_threshold}, sigma_rho={sigma_rho})."
            )
            det_score_full, p_det_array = compute_det_score(
                events,
                snr,
                snr_threshold=snr_threshold,
                sigma_rho=sigma_rho,
                p_det_min=p_det_min,
            )
            # Restrict det_score columns to the phys_keys this model uses
            columns = [POPULATION_KEYS.index(key) for key in phys_keys]
            det_score = det_score_full[:, columns]
            near_threshold = int((p_det_array < 0.99).sum())
            logging.info(
                f"p_det range: [{p_det_array.min():.4f}, {p_det_array.max():.4f}]; "
                f"events with p_det < 0.99 (near threshold, non-zero gradient): "
                f"{near_threshold} / {number_of_events}. "
                f"Term IV will be "
                f"{'negligible' if near_threshold == 0 else 'non-zero'}."
            )

    # Term III importance weight 1/p_det(theta_k), floored at p_det_min so
    # that an event far below threshold cannot blow up the sum.
    if p_det_per_event is None:
        logging.warning(
            "No per-event detection probability available (pass `snr` or "
            "`p_det_per_event`); assuming p_det(theta_k) = 1 for Term III, "
            "i.e. a complete catalogue."
        )
        inverse_p_det = numpy.ones(number_of_events)
    else:
        p_det_per_event = numpy.asarray(p_det_per_event)
        inverse_p_det = 1.0 / numpy.maximum(p_det_per_event, p_det_min)
        logging.info(
            f"Term III weight 1/p_det(theta): "
            f"min={inverse_p_det.min():.4f}, max={inverse_p_det.max():.4f}, "
            f"mean={inverse_p_det.mean():.4f}"
        )

    # ------------------------------------------------------------------
    # Lambda-Hessians of the four scalar building blocks
    # ------------------------------------------------------------------
    logging.info("Computing Terms II-V (correction terms)...")
    d2_logdet, d2_trace, d2_termV, d2_termIV = _hessians_correction_terms_fd(
        model,
        events,
        fim_sub,
        free_names,
        params_vec,
        phys_keys,
        h_lam=h_lam,
        h_theta=h_theta,
        detection_score=det_score,
    )

    def _nansum_terms(d2, weights=None):
        """Sum over events with optional weights, NaN contributions as zero."""
        if d2 is None:
            return numpy.zeros((number_of_free, number_of_free))
        if weights is not None:
            d2 = d2 * weights
        return numpy.nansum(d2, axis=-1)

    # Weights follow Gair+2022 Eq. 21 -- see the Notes above.  Terms I, II and
    # V carry P_det(theta)/P_det(lambda), which the detected catalogue already
    # samples, so they need no explicit weight.  Term III carries
    # 1/P_det(lambda) only, which becomes 1/p_det(theta_k) per event.  Term IV
    # likewise, but its p_det(theta_k) cancels against D_l = p_det * det_score.
    return CorrectionTerms(
        fisher_II=+0.5 * _nansum_terms(d2_logdet),
        fisher_III=-0.5 * _nansum_terms(d2_trace, weights=inverse_p_det),
        fisher_IV=-1.0 * _nansum_terms(d2_termIV),
        fisher_V=-0.5 * _nansum_terms(d2_termV),
        p_det=p_det_per_event,
        detection_score=det_score,
    )
