import os
import json
import numpy as np
from math import pi
from pymatgen.analysis.diffraction.core import get_unique_families

with open(os.path.join(os.path.dirname(__file__), 'neutron_scattering_length.json')) as file:
    atomic_scattering_length = json.load(file)

class NeutronDiffraction():
    def __init__(self, wavelength = 1, q_tol = 1e-5, relative_intensity_tol = 1e-5):
        self.wavelength = wavelength
        self.q_tol = q_tol
        self.relative_intensity_tol = relative_intensity_tol

    def get_pattern(self, structure):
        wavelength = self.wavelength
        latt = structure.lattice

        min_r, max_r = 0, 2 / wavelength

        recip_latt = latt.reciprocal_lattice_crystallographic
        recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_r)
        if min_r:
            recip_pts = [pt for pt in recip_pts if pt[1] >= min_r]

        try:
            fs = np.array(list(map(lambda x: atomic_scattering_length[x.symbol], structure.species)))
        except KeyError:
            raise ValueError('Unknown atomic scattering length')
        frac_coords = structure.frac_coords

        peaks = {}
        qs = []

        for hkl, q, ind, _ in sorted(recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])):
            hkl = [int(round(i)) for i in hkl]
            if q != 0:

                q_dot_r = np.dot(frac_coords, np.transpose([hkl])).T[0]
                F_hkl = np.sum(fs * np.exp(2j * pi * q_dot_r))
                I_hkl = (F_hkl * F_hkl.conjugate()).real

                ind = np.where(np.abs(np.subtract(qs, q)) < self.q_tol)[0]
                if len(ind) > 0:
                    peaks[qs[ind[0]]][0] += I_hkl
                    peaks[qs[ind[0]]][1].append(tuple(hkl))
                else:
                    peaks[q] = [I_hkl, [tuple(hkl)]]
                    qs.append(q)

        max_intensity = max(peak[0] for peak in peaks.values())
        qs = []
        Is = []
        hklss = []

        for key in sorted(peaks):
            peak = peaks[key]
            fam = get_unique_families(peak[1])
            if peak[0] / max_intensity > self.relative_intensity_tol:
                qs.append(key)
                Is.append(peak[0])

                hkls = []
                for hkl, mult in fam.items():
                    hkls += [hkl]*mult
                hklss.append(hkls)

        return Is, hklss, fs, frac_coords, qs