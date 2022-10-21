#!/usr/bin/env python
"""Functions for the computation of twist."""
import math

import numba
import numba.experimental.jitclass as jitclass
import numpy as np
import pysmsh.interpolate.staggered
import pysmsh.interpolate.trilinear
import scipy.constants

def calc_vhat(x1, y1, z1, x2, y2, z2):
    """
    Calculate the unit vector from point 1 (x1, y1, z1) to point 2 (x2, y2, z2).
    """

    vx, vy, vz = x2-x1, y2-y1, z2-z1
    vabs = np.sqrt(vx**2 + vy**2 + vz**2)
    vhat = vx/vabs, vy/vabs, vz/vabs

    return vhat

def find_V(axis_crds, fl_crds, Bhat, interval, rotsens):
    """
    Find the points from fl_crds closest to the normal of each point of axis_crds.

    Parameters
    ----------
    axis_crds : (3,X) list of floats
        Coordinates of the axis field line.
    fl_crds : (3,Y) list of floats
        Coordinates of the field line to be searched. Must have Y > X.
    Bhat : (X,3) list of floats
        The unit tangent vectors for each point of the axis field line.
    interval : int
        The search interval in terms of `fl_crds` indices.
    rotsens
        The rotation sensitivity angle in degrees. Successive matches must
        not rotate more than `rotsens`.

    Returns
    -------
    Vhat : (N,Z,3) list of floats
        Unit normal vectors for each point of the axis field line to the other field line.
        Ideally Z = X, but the search may be aborted early.
    BdV : (N,Z) list of floats
        Dot products between `Bhat` and `Vhat`.
    indexes : (N,Z) list of int
        The index of `fl_crds` matched for each point of `axis_crds`.

    Notes
    -----
    Vhat will be returned as all NaNs if a spike in BdV is detected in the upper
    80% of the axis coordinates. This suggests it matched to the wrong part of the
    other field line and/or that the input should be adjusted.

    In general, `fl_crds` should be at a higher resolution than `axis_crds`.

    N = the number of field lines.
    """
    ppoint = 0 # Index of fl_crds found for the previous axis_crds.
    start = 0 # Index for starting the search.
    end = start + interval
    max_index = len(fl_crds[0]) - interval - 1
    BdV = [] # Final list of BdotV values 1:1 with axis_crds.
    Vhat = [] # Final list of Vhat values 1:1 with axis_crds.
    indexes = [] # Store the matched indexes for debugging.
    # For each axis coordinate.
    for i in range(len(axis_crds[0])):
        end = start + 2*interval
        if end > (max_index + interval): end = max_index + interval
        if start == len(fl_crds[0]):
            print("The search interval was reduced to 0 and cannot be extended. Search aborted at axis index "+str(i)+" of "+str(len(axis_crds[0])-1))
            break
        if start == end:
            print("The search interval was reduced to 0. Adding 1.")
            end += 1

        tBdV = [] # Temporary storage for BdotVs.
        tVhat = [] # Temporary storage for Vhats.
        for j in range(start, end):
            cVhat = calc_vhat(axis_crds[0][i], axis_crds[1][i], axis_crds[2][i],
                              fl_crds[0][j], fl_crds[1][j], fl_crds[2][j])
            tBdV.append((Bhat[i][0] * cVhat[0]) + (Bhat[i][1] * cVhat[1]) + (Bhat[i][2] * cVhat[2]))
            tVhat.append(cVhat)

        found = 0

        while found == 0:
            BdVmin = min(tBdV, key=lambda x:abs(x))
            index_BdVmin = tBdV.index(BdVmin)
            if i == 0 or i in range(int(len(axis_crds[0])/10)) or i == len(axis_crds[0])-1: break
            # Calculate previous dot current.
            pdc = (Vhat[i-1][0] * tVhat[index_BdVmin][0]) + (Vhat[i-1][1] * tVhat[index_BdVmin][1]) + (Vhat[i-1][2] * tVhat[index_BdVmin][2])
            if abs(math.degrees(math.acos(pdc))) > rotsens:
                del tBdV[index_BdVmin]
                del tVhat[index_BdVmin]
                if len(tBdV) == 0:
                    break
            else:
                found = 1

        if len(tBdV) == 0:
            print("Search aborted because the rotation angle condition cannot be satisfied at index "+str(i)+". Increasing rotsens may be required.")
            break
        Vhat.append(tVhat[index_BdVmin])
        BdV.append(BdVmin)

        ppoint = start + tBdV.index(BdVmin)
        indexes.append(ppoint)
        start = ppoint + 1
        if start >= max_index: start = max_index
        if start <= ppoint: start = ppoint + 1

    # Check for cases with a spike in B dot V, indicative of belonging to another structure
    # or potentially bad search parameters.
    # Search the upper 80% of the field line due to spikes at footpoints.
    if len(axis_crds[0]) < 10:
        print("This result may be less reliable due to the low number of axis_crds. Increasing the resolution is advised.")
    else:
        step = int(len(axis_crds[0])/10)
        if abs(max(BdV[step:-step], key=abs)) > 0.01:
            Vhat = [(float('nan'), float('nan'), float('nan')) for x in Vhat]
            print("Returning Vhat of NaNs due to suspected spike in Bhat dot Vhat.")

    return Vhat, BdV, indexes

def calc_s(fl_crds):
    """
    Calculates the arc length of each point along the field line corresponding to
    the input coordinates fl_crds, compared to the first point.

    Parameters
    ----------
    fl_crds : (3,X) list of floats
        Coordinates of the field line.

    Returns
    -------
    s_arc : (X) list of floats
        The arc length for each point along the field line compared to the first
        point defined as 0.
    """
    s_arc = []
    for i in range(len(fl_crds[0])):
        # For the start of the arc.
        if i == 0:
            s_arc.append(0)
            continue
        # For the rest.
        s = s_arc[i-1] + np.sqrt((fl_crds[0][i] - fl_crds[0][i-1])**2 +
                                 (fl_crds[1][i] - fl_crds[1][i-1])**2 +
                                 (fl_crds[2][i] - fl_crds[2][i-1])**2)
        s_arc.append(s)

    return s_arc

# Need B-hat dot V-hat cross dV-hat/ds for each point.
# Note that Vhats are sometimes shorter than axis_crds because aborted when interval = 0.
def calc_dVds(Vhat, s_arc):
    """
    Calculates differencing for Vhat using the 1D coordinates of s_arc.

    Parameters
    ----------
    Vhat : (X,3) list of floats
        Unit normal vectors for each point of the axis field line to the other field line.
        Generally provided by the find_V function.
    s_arc : (Y) list of floats
        The arc length for each point along the field line compared to the first
        point defined as 0. Generally provided by the calc_s function.

    Returns
    -------
    dVds : (X) list of floats
        The differencing of `Vhat` using the 1D coordinates of `s_arc`.
    dV : (X) list of floats
        The differencing of `Vhat` used for diagnostic purposes.
    ds : (X) list of floats
        The change in `s_arc` used for diagnostic purposes.

    Notes
    -----
    The length of `Vhat` may be shorter than `s_arc`, but not the other way around.
    """
    dV = []
    ds = []
    dVds = []
    for i in range(len(Vhat)):
        # Forward difference
        if i == 0:
            tds = s_arc[i+1] - s_arc[i]
            tdV = ((Vhat[i+1][0]) - (Vhat[i][0]),
                   (Vhat[i+1][1]) - (Vhat[i][1]),
                   (Vhat[i+1][2]) - (Vhat[i][2]))
            tdVds = (tdV[0]/tds, tdV[1]/tds, tdV[2]/tds)
            dV.append(tdV)
            ds.append(tds)
            dVds.append(tdVds)
            continue
        # Backward difference
        if i == len(Vhat)-1:
            tds = s_arc[i] - s_arc[i-1]
            tdV = ((Vhat[i][0] - Vhat[i-1][0]),
                   (Vhat[i][1] - Vhat[i-1][1]),
                   (Vhat[i][2] - Vhat[i-1][2]))
            tdVds = (tdV[0]/tds, tdV[1]/tds, tdV[2]/tds)
            dV.append(tdV)
            ds.append(tds)
            dVds.append(tdVds)
            continue
        # Central difference
        tds = s_arc[i+1] - s_arc[i-1]
        tdV = ((Vhat[i+1][0] - Vhat[i-1][0]),
               (Vhat[i+1][1] - Vhat[i-1][1]),
               (Vhat[i+1][2] - Vhat[i-1][2]))
        tdVds = (tdV[0]/tds, tdV[1]/tds, tdV[2]/tds)
        dV.append(tdV)
        ds.append(tds/2)
        dVds.append(tdVds)

    return dVds, dV, ds

def calc_Tg(Bhat, Vhat, axis_crds, ds_in=None):
    """
    Calculates Tg for the input field line(s) at the input axis.

    Parameters
    ----------
    Bhat : (X,3) list of floats
        The unit tangent vectors for each point of the axis field line.
    Vhat : (N,Y,3) list of floats
        Unit normal vectors for each point of the axis field line to the field line(s)
        of interest. Generally provided by the find_V function.
    axis_crds : (3,X) list of floats
        Coordinates of the axis field line.
    ds_in : (X) list of floats, optional
        The arc length for each point along the axis field line compared to the
        first point defined as 0. Will be computed from `axis_crds` by the calc_s
        function if not provided.

    Returns
    -------
    Tg : (N) list of floats
        Tg for the field line corresponding to `Vhat`.
    debugTg : (N,Y) list of floats
        The contributions of each step along the axis field line to the total
        Tg integral.
    VcdVds : (N,Y,3) list of floats
        Vhat cross dV/ds for each index of Vhat used for diagnostic purposes.

    Notes
    -----
    `axis_crds` and `Bhat` must be the same length. `Vhat` may be shorter but it
    must correspond to `axis_crds` and `Bhat`, with index 0 being a common start.

    N = the number of field lines.
    """
    # Compute dV/ds
    dVds = []
    dV = []
    ds = []
    s_arc = calc_s(axis_crds)
    for i in range(len(Vhat)):
        tdVds, tdV, tds = calc_dVds(Vhat[i], s_arc)
        dVds.append(tdVds)
        dV.append(tdV)
        ds.append(tds)
    if ds_in is not None: ds = ds_in
    # Compute V-hat cross dVds
    VcdVds = []
    for m in range(len(Vhat)):
        temp_n = [] # temporary list for n loop
        for n in range(len(Vhat[m])):
            tVcdVds = (((Vhat[m][n][1] * dVds[m][n][2]) - (Vhat[m][n][2] * dVds[m][n][1])),
                       ((Vhat[m][n][2] * dVds[m][n][0]) - (Vhat[m][n][0] * dVds[m][n][2])),
                       ((Vhat[m][n][0] * dVds[m][n][1]) - (Vhat[m][n][1] * dVds[m][n][0])))
            temp_n.append(tVcdVds)
        VcdVds.append(temp_n)
    # Compute ((B-hat dot VcdVds) * ds)/(2 * pi) and sum to give Tg
    Tg = []
    Tg_comps = []
    for p in range(len(VcdVds)):
        tTg = 0
        temp_Tgcomps = []
        for q in range(len(VcdVds[p])):
            ttTg = (((Bhat[q][0] * VcdVds[p][q][0]) +
                     (Bhat[q][1] * VcdVds[p][q][1]) +
                     (Bhat[q][2] * VcdVds[p][q][2])) * ds[p][q]) / (2*np.pi)
            tTg += ttTg
            temp_Tgcomps.append(ttTg)
        Tg_comps.append(temp_Tgcomps)
        Tg.append(tTg)

    return Tg, Tg_comps, VcdVds

@jitclass()
class TwCompute:

    magnetic_field : pysmsh.interpolate.staggered.FaceStaggeredInterpolator
    current_density : pysmsh.interpolate.trilinear.EdgeStaggeredInterpolator

    def __init__(self, magnetic_field_interpolator, current_density_interpolator):
        self.magnetic_field = magnetic_field_interpolator
        self.current_density = current_density_interpolator

    def compute(self, p):

        # Magnetic field vector
        B = self.magnetic_field.value(p)

        # Current density vector
        J = self.current_density.value(p)

        # B dot B = |B|^2 = B^2
        Bsqr = B[0]**2 + B[1]**2 + B[2]**2

        integrand = 0.0

        if Bsqr > 0.0:

            # Integrand of Tw path integral (ignoring the constant factor mu0/4 pi)
            #  = J_parallel / |B| = J dot b / |B| = (J dot B) / B^2
            J_dot_B = J[0]*B[0] + J[1]*B[1] + J[2]*B[2]

            integrand = J_dot_B/Bsqr

        return integrand

@numba.njit()
def compute_Tw_map(coords1, coords2, coords3, order, b_field_interpolator, tracer, Tw_integrator, lower_boundary):
    """Computes Tw on a 2D plane that intercepts the third axis at the index given.

    Input:
    x/y/z_crds: Two must be a coordinate grid, the third must be the index to cut that axis.
    """
    Tw_map = np.zeros((len(coords1), len(coords2)))
    is_closed = np.zeros((len(coords1), len(coords2)))

    in_point = np.zeros(3)

    list_of_seed_points = list()

    for i, x in enumerate(coords1):
        for j, y in enumerate(coords2):

            in_point[order[0]] = x
            in_point[order[1]] = y
            in_point[order[2]] = coords3

            # Trace in the direction of the field.
            tracer.follow_field_direction = True
            start_pt = tracer.compute(in_point, b_field_interpolator, Tw_integrator)
            Tw1 = tracer.integral

            # Trace in the opposite direction of the field.
            tracer.follow_field_direction = False
            end_pt = tracer.compute(in_point, b_field_interpolator, Tw_integrator)
            Tw2 = tracer.integral

            if (start_pt[2] <= lower_boundary) and (end_pt[2] <= lower_boundary):
                is_closed[i, j] = 1.0

            Tw_map[i, j] = Tw1 + Tw2

            list_of_seed_points.append(np.copy(in_point))

    Tw_map *= scipy.constants.mu_0/(4.0*np.pi)

    return Tw_map, is_closed, list_of_seed_points
