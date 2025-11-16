"""Repackaged and slightly modified modified version of the empymod
tutorial from here:
https://empymod.emsig.xyz/en/stable/gallery/tdomain/tem_walktem.html

waveform - From empymod tutorial: apply a TEM waveform to a modeled sounding
get_time - From empymod tutorial: determine necessary time padding for TEM model
walktem - Modified from empymod tutorial: run a TEM forward model
"""

import configparser
import glob
import os
import sys

import empymod
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.special import roots_legendre

def split_x(x):
    # Split inversion parameter vector into resistivities and thicknesses
    nlyr = (len(x)+1)//2
    res = x[:nlyr]
    thk = x[nlyr:]
    return res, thk

def strat_plot(thick, res, hatches=['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'], vmin=0, vmax=500, cmap="RdBu", colors=None, labels=None):
    """Utility function to plot a resistivity column."""
    thick = thick[:]
    
    # Add bottom layer to thick for plotting
    if(len(thick) == 0):
        thick = [1]
    else:
        thick.append(.2*np.sum(thick))

    # Set up colormap
    cNorm  = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    
    # Make plot
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    for i, h in enumerate(thick):
        if(colors is not None and type(colors) is list):
            color = colors[i % len(colors)]
        else:
            color = scalarMap.to_rgba(res[i])
        ax.bar([0], [h], bottom=np.sum(thick[:i]), width=1, hatch=hatches[i % len(hatches)], color=color, edgecolor="k")


    ax.set_ylim([np.sum(thick), 0]) # tight top and bottom margins, zero at the top
    for spine in ["bottom", "top", "left", "right"]:
        ax.spines[spine].set_visible(False)

    ax.get_xaxis().set_visible(False) # No x axis ticks

    # Set y axis ticks
    ticks = np.append(0, np.cumsum(thick)[:-1])
    tick_labels = ["%.1f m" % tick for tick in ticks]
    ax.get_yaxis().set_ticks(ticks)
    ax.get_yaxis().set_ticklabels(tick_labels) 
    
    if(colors is not None and type(colors) is list):
        labels = ["%.1f" % r for r in res]
        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), title = "Resistivity (ohm-m)", handlelength=3, handleheight=3, frameon=False)
    else:
        plt.colorbar(scalarMap, ax=ax, label="Resistivity (ohm m)")

def waveform(times, resp, times_wanted, wave_time, wave_amp, nquad=3):
    """Apply a source waveform to the signal.

    Parameters
    ----------
    times : ndarray
        Times of computed input response; should start before and end after
        `times_wanted`.

    resp : ndarray
        EM-response corresponding to `times`.

    times_wanted : ndarray
        Wanted times.

    wave_time : ndarray
        Time steps of the wave.

    wave_amp : ndarray
        Amplitudes of the wave corresponding to `wave_time`, usually
        in the range of [0, 1].

    nquad : int
        Number of Gauss-Legendre points for the integration. Default is 3.

    Returns
    -------
    resp_wanted : ndarray
        EM field for `times_wanted`.

    """

    # Interpolate on log.
    PP = iuSpline(np.log10(times), resp)

    # Wave time steps.
    dt = np.diff(wave_time)
    dI = np.diff(wave_amp)
    dIdt = dI / dt

    # Gauss-Legendre Quadrature; 3 is generally good enough.
    # (Roots/weights could be cached.)
    g_x, g_w = roots_legendre(nquad)

    # Pre-allocate output.
    resp_wanted = np.zeros_like(times_wanted)

    # Loop over wave segments.
    for i, cdIdt in enumerate(dIdt):
        # We only have to consider segments with a change of current.
        if cdIdt == 0.0:
            continue

        # If wanted time is before a wave element, ignore it.
        ind_a = wave_time[i] < times_wanted
        if ind_a.sum() == 0:
            continue

        # If wanted time is within a wave element, we cut the element.
        ind_b = wave_time[i + 1] > times_wanted[ind_a]

        # Start and end for this wave-segment for all times.
        ta = times_wanted[ind_a] - wave_time[i]
        tb = times_wanted[ind_a] - wave_time[i + 1]
        tb[ind_b] = 0.0  # Cut elements

        # Gauss-Legendre for this wave segment. See
        # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        # for the change of interval, which makes this a bit more complex.
        logt = np.log10(np.outer((tb - ta) / 2, g_x) + (ta + tb)[:, None] / 2)
        fact = (tb - ta) / 2 * cdIdt
        resp_wanted[ind_a] += fact * np.sum(np.array(PP(logt) * g_w), axis=1)

    return resp_wanted


def get_time(time, r_time):
    """Additional time for ramp.

    Because of the arbitrary waveform, we need to compute some times before and
    after the actually wanted times for interpolation of the waveform.

    Some implementation details: The actual times here don't really matter. We
    create a vector of time.size+2, so it is similar to the input times and
    accounts that it will require a bit earlier and a bit later times. Really
    important are only the minimum and maximum times. The Fourier DLF, with
    `pts_per_dec=-1`, computes times from minimum to at least the maximum,
    where the actual spacing is defined by the filter spacing. It subsequently
    interpolates to the wanted times. Afterwards, we interpolate those again to
    compute the actual waveform response.

    Note: We could first call `waveform`, and get the actually required times
          from there. This would make this function obsolete. It would also
          avoid the double interpolation, first in `empymod.model.time` for the
          Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
          Probably not or marginally faster. And the code would become much
          less readable.

    Parameters
    ----------
    time : ndarray
        Desired times

    r_time : ndarray
        Waveform times

    Returns
    -------
    time_req : ndarray
        Required times
    """
    tmin = np.log10(max(time.min() - r_time.max(), 1e-10))
    tmax = np.log10(time.max() - r_time.min())
    return np.logspace(tmin, tmax, time.size + 2)


def walktem(res, thick, off_times, tx_waveform, tx_side):
    """Custom wrapper of empymod.model.bipole.

    Here, we compute WalkTEM data using the ``empymod.model.bipole`` routine as
    an example. We could achieve the same using ``empymod.model.dipole`` or
    ``empymod.model.loop``.

    We model the big source square loop by computing only half of one side of
    the electric square loop and approximating the finite length dipole with 3
    point dipole sources. The result is then multiplied by 8, to account for
    all eight half-sides of the square loop.

    The implementation here assumes a central loop configuration, where the
    receiver (1 m2 area) is at the origin, and the source is a tx_side x tx_side m electric
    loop, centered around the origin.


    Parameters
    ----------
    moment : str {'lm', 'hm'}
        Moment. If 'lm', above defined ``lm_off_time``, ``lm_waveform_times``,
        and ``lm_waveform_current`` are used. Else, the corresponding
        ``hm_``-parameters.

    res : ndarray
        Resistivities of the resistivity model (see ``empymod.model.bipole``
        for more info.)

    thick : ndarray
        Depths of the resistivity model (see ``empymod.model.bipole`` for more
        info.)

    off_times : ndarray
        Receive gate times.

    tx_waveform : dict {"i": ndarray, "t": ndarray}
        Dictionary describing the transmit waveform. Must have two items of equal length,
        "i", an array describing the current levels and "t", an array giving the time coordinates
        of the current changes described in i.

    tx_side : float
        Side length of the square transmit loop

    Returns
    -------
    WalkTEM : EMArray
        WalkTEM response (dB/dt).

    """
    # Add extra off_time to pad response
    off_times = np.append(off_times, off_times[-1] + (off_times[-1]-off_times[-2]))

    # Thickness -> depth
    depth = np.cumsum(thick)
    
    # === GET REQUIRED TIMES ===
    time = get_time(off_times, tx_waveform["t"])

    # === GET REQUIRED FREQUENCIES ===
    time, freq, ft, ftarg = empymod.utils.check_time(
        time=time,  # Required times
        signal=1,  # Switch-on response
        ft="dlf",  # Use DLF
        ftarg={"dlf": "key_81_2009"},  # Short, fast filter; if you
        verb=0,  # need higher accuracy choose a longer filter.
    )

    # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
    # We only define a few parameters here. You could extend this for any
    # parameter possible to provide to empymod.model.bipole.
    EM = empymod.model.bipole(
        src=[
            tx_side / 2,
            tx_side / 2,
            0,
            tx_side / 2,
            0,
            0,
        ],  # El. bipole source; half of one side.
        rec=[0, 0, 0, 0, 90],  # Receiver at the origin, vertical.
        depth=np.r_[0, depth],  # Depth-model, adding air-interface.
        res=np.r_[2e14, res],  # Provided resistivity model, adding air.
        freqtime=freq,  # Required frequencies.
        mrec=True,  # It is an el. source, but a magn. rec.
        strength=8,  # To account for 4 sides of square loop.
        srcpts=3,  # Approx. the finite dip. with 3 points.
        htarg={"dlf": "key_101_2009"},  # Short filter, so fast.
        verb=0,
    )

    # Multiply the frequecny-domain result with
    # \mu for H->B, and i\omega for B->dB/dt.
    EM *= 2j * np.pi * freq * 4e-7 * np.pi

    # === Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)===
    # Note: Here we just apply one filter. But it seems that WalkTEM can apply
    #       two filters, one before and one after the so-called front gate
    #       (which might be related to ``delay_rst``, I am not sure about that
    #       part.)
    cutofffreq = 4.5e5  # As stated in the WalkTEM manual
    h = (1 + 1j * freq / cutofffreq) ** -1  # First order type
    h *= (1 + 1j * freq / 3e5) ** -1
    EM *= h

    # === CONVERT TO TIME DOMAIN ===
    delay_rst = 1.8e-7  # As stated in the WalkTEM manual
    # delay_rst -= 1.830e-6
    # factor = 1.027
    EM, _ = empymod.model.tem(
        EM[:, None], np.array([1]), freq, time + delay_rst, 1, ft, ftarg
    )
    EM = np.squeeze(EM)

    # === APPLY WAVEFORM, trim off pad sample ===
    return waveform(time, EM, off_times, tx_waveform["t"], tx_waveform["i"])[:-1]
