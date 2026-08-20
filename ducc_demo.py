import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, curvedsky, wcsutils, enplot
from ducc0 import sht

import argparse
import os
from os.path import join as opj

# load the ACT DR6 map, provide the directory in which you downloaded the map
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str,
                    help='The path of the directory on disk where you '
                    'downloaded the ACT DR6 coadd map.')
parser.add_argument('--lmax', type=int, default=10000,
                    help='The bandlimit (maximum harmonic multipole) to use.')
parser.add_argument('--nthreads', type=int, default=0,
                    help='The number of CPU threads to use. If 0, the '
                    'detected (by ducc) number of threads available.')
args = parser.parse_args()

data_dir = args.dir
lmax = args.lmax
nthreads = args.nthreads

# we will use pixell utilities just to load the data and wcs from the file.
# while pixell also wraps ducc0 to handle general calls to spherical
# harmonic transforms, we won't use them, instead exposing the underlying
# ducc0 functions
imap = enmap.read_fits(opj(data_dir, 'act-planck_dr4dr6_coadd_AA_daynight_f090_map.fits'))

# show that imap has 3 Stokes components -- intensity (I) and polarization (Q, U),
# a given shape, is single-precision, and its wcs information (pixel resolution and
# reference location). finally, show that the rectangular pixelization scheme maps
# onto one of the geometries mentioned here:
# * https://mtr.pages.mpcdf.de/ducc/sht.html#ducc0.sht.get_gridweights
# * https://arxiv.org/abs/1303.4945
# for which there exist exact quadrature weights (i.e., exact SHT analysis up to a
# given lmax is possible)
shape, wcs = imap.geometry
ducc_geometry = curvedsky.get_ducc_geo(shape=imap.shape, wcs=imap.wcs)

print(f'Map shape: {shape}')
print(f'Map dtype: {imap.dtype}')
print(f'Map wcs: {wcs}')
assert ducc_geometry.name == 'F1', f'Expected F1, got {ducc_geometry.name=}'
print(f'Map quadrature scheme: {ducc_geometry.name}')

# first we will compute the spherical harmonic coefficients (the "alm" values)
# via a spherical harmonic analysis (map --> alm). we will do so inside some
# smooth mask that cuts the very-noisy map edges and the galaxy. this is for 
# realism, but has nothing to do with whether or not a transform is possible.

# load the mask and confirm it has the same shape and wcs as the data (so
# they can be sensibly multiplied together)
mask = enmap.read_fits(opj(data_dir, 'window_dr6_pa6_f090_kspace.fits'))

assert mask.shape == imap.shape[-2:], \
    f'Expected same shape in last two dimensions, got ' + \
    f'{mask.shape=} and {imap.shape[-2:]=}'
assert wcsutils.equal(mask.wcs, imap.wcs), \
    f'Expected same wcs in last two dimensions, got ' + \
    f'{mask.wcs=} and {imap.wcs=}'

masked_imap = mask * imap

# save plots of the raw and masked maps so we can gain intuition for the data
for i in range(3):
    p = enplot.plot([imap[i], masked_imap[i]], downgrade=32, colorbar=True, ticks=15)
    enplot.write(opj(data_dir, f"act_dr6_planck_coadd_{'IQU'[i]}"), p)

# our data and ducc assume a different convention for the ordering of rings/rows
# (iso-latitude) and columns (iso-longitude). ducc assumes rows/columns decrease/increase
# their declination/right-ascension, whereas our data is the opposite. so we need to
# flip our data in both axes. the wcs of the flipped data is automatically updated
flipped_masked_imap = masked_imap[..., ::-1, ::-1]

# the map pixelization does not span the fullsky, instead it has a "cut" declination
# range from about -63 degrees to +23 degrees. this is because the ACT data only span
# this range, so we save disk/memory. this affects how we ducc does the spherical harmonic
# analysis:
# 1. We can use sht.adjoint_synthesis. We need to give ducc more info about the
#    declination rings; in principle these don't have to follow one of the above
#    geometries with exact quadrature weights (but in this case, they do)
# 2. We can use sht.adjoint_synthesis_2d if we first "insert" the cut map into a
#    a map that spans the full declination range. We only specify the quadrature
#    scheme which is 'F1' in this case. The pixelization must follow one of the 
#    quadrature schemes

###############################################################################
############################### Do 1st method #################################

# in this case we also need to multiply by the quadrature weights. the weights
# are for the full dec range, so we need to slice-out the exact rows for our map
fullsky_quadw = sht.get_gridweights(ducc_geometry.name, ducc_geometry.ny)
fullsky_quadw /= ducc_geometry.nx
fullsky_quadw = fullsky_quadw.astype(imap.dtype, copy=False)
quadw = fullsky_quadw[ducc_geometry.yoff:ducc_geometry.yoff + imap.shape[-2]]

# get necessary arguments for ducc. these will be the co-latitudes of 
# each ring as an array in 'theta', the number of columns in each ring
# in 'nphi', the starting longitude of each ring in 'phi0', and the
# index into the flattened 2d map where each ring starts in 'ringstart'
ducc_rings = curvedsky.get_ring_info(*flipped_masked_imap.geometry)

# perform transform (spin-0 and spin-2). here is where you would try a new
# implementation and compare results!
alm_T = sht.adjoint_synthesis(
    map=(quadw[..., None] * flipped_masked_imap)[0:1].reshape(1, -1), spin=0,
    lmax=lmax, theta=ducc_rings.theta, nphi=ducc_rings.nphi,
    phi0=ducc_rings.phi0, ringstart=ducc_rings.offsets,
    nthreads=nthreads
    )

alm_EB = sht.adjoint_synthesis(
    map=(quadw[..., None] * flipped_masked_imap)[1:].reshape(2, -1), spin=2, 
    lmax=lmax, theta=ducc_rings.theta, nphi=ducc_rings.nphi,
    phi0=ducc_rings.phi0, ringstart=ducc_rings.offsets,
    nthreads=nthreads
    )

alm_TEB = np.concatenate([alm_T, alm_EB], axis=0)

np.save(opj(data_dir, 'ducc_alm_TEB_adjoint_synthesis.npy'), alm_TEB)

###############################################################################
###############################################################################

###############################################################################
############################### Do 2nd method #################################

# NOTE: no quadrature weights manually applied, ducc supplies them internally

# need to insert the data into a full-sky array first.
# NOTE: here, i am manually inserting what i know to be the correct res and variant.
# this just gives me an array with the correct shape and wcs for a fullsky map
fullsky_masked_imap_shape, fullsky_masked_imap_shape_wcs = enmap.fullsky_geometry(
    res=np.deg2rad(1/120), variant='fejer1'
    )

flipped_fullsky_masked_imap = enmap.zeros(
    shape=(3, *fullsky_masked_imap_shape),
    wcs=fullsky_masked_imap_shape_wcs,
    dtype=masked_imap.dtype
    )[..., ::-1, ::-1]

flipped_fullsky_masked_imap[..., ducc_geometry.yoff:ducc_geometry.yoff + imap.shape[-2], :] = flipped_masked_imap

# get necessary arguments for ducc. this is just the single value giving
# the longitude where all rings start in 'phi0'. unlike the 1st method,
# this method enforces a regular rectangular pixelization where all rings
# are aligned in longitude, so we just need one number
ducc_info = curvedsky.analyse_geometry(*flipped_fullsky_masked_imap.geometry)

# perform transform (spin-0 and spin-2). here is where you would try a new
# implementation and compare results!
alm_T = sht.analysis_2d(
    map=(flipped_fullsky_masked_imap)[0:1], spin=0,
    lmax=lmax, geometry=ducc_geometry.name,
    phi0=ducc_info.phi0,
    nthreads=nthreads
    )

alm_EB = sht.analysis_2d(
    map=(flipped_fullsky_masked_imap)[1:], spin=2,
    lmax=lmax, geometry=ducc_geometry.name,
    phi0=ducc_info.phi0,
    nthreads=nthreads
    )

alm_TEB = np.concatenate([alm_T, alm_EB], axis=0)

np.save(opj(data_dir, 'ducc_alm_TEB_analysis_2d.npy'), alm_TEB)

###############################################################################
###############################################################################

# now do synthesis operation (alm --> map). because the analysis operation is
# bandlimited, the synthesized map will not be equivalent to the analyzed map;
# instead it will scales with multipole higher than the bandlimit (i.e., 
# with physically smaller wavelengths). as before, we can do this in two ways:
# 1. We can use sht.synthesis. We need to give ducc more info about the
#    declination rings; in principle these don't have to follow one of the above
#    geometries with exact quadrature weights (but in this case, they do)
# 2. We can use sht.synthesis_2d if we "extract" the cut map out of a
#    a map that spans the full declination range. We only specify the quadrature
#    scheme which is 'F1' in this case. The pixelization must follow one of the 
#    quadrature schemes

###############################################################################
############################### Do 1st method #################################

# NOTE: quadrature weights only needed for analysis, not synthesis!

# get necessary arguments for ducc. these will be the co-latitudes of 
# each ring as an array in 'theta', the number of columns in each ring
# in 'nphi', the starting longitude of each ring in 'phi0', and the
# index into the flattened 2d map where each ring starts in 'ringstart'
ducc_rings = curvedsky.get_ring_info(*flipped_masked_imap.geometry)

# perform transform (spin-0 and spin-2). here is where you would try a new
# implementation and compare results!
map_I = sht.synthesis(
    alm=alm_T, spin=0,
    lmax=lmax, theta=ducc_rings.theta, nphi=ducc_rings.nphi,
    phi0=ducc_rings.phi0, ringstart=ducc_rings.offsets,
    nthreads=nthreads
    )

map_QU = sht.synthesis(
    alm=alm_EB, spin=2, 
    lmax=lmax, theta=ducc_rings.theta, nphi=ducc_rings.nphi,
    phi0=ducc_rings.phi0, ringstart=ducc_rings.offsets,
    nthreads=nthreads
    )

map_IQU = np.concatenate([map_I, map_QU], axis=0)

# now we need to reshape the map because its last axis is 1d, and 
# flip it back to the original declination and right-ascension ordering.
# finally, reassign original wcs information and write to disk.
map_IQU = map_IQU.reshape(*flipped_masked_imap.shape)
map_IQU = map_IQU[..., ::-1, ::-1]
map_IQU = enmap.ndmap(map_IQU, masked_imap.wcs)

enmap.write_map(opj(data_dir, 'ducc_map_IQU_synthesis.fits'), map_IQU)

###############################################################################
###############################################################################

# plot to visualize the synethized map and compare to the original map
for i in range(3):
    p = enplot.plot(map_IQU[i], downgrade=32, colorbar=True, ticks=15)
    p_diff = enplot.plot(map_IQU[i] - masked_imap[i], downgrade=32, colorbar=True, ticks=15)

    enplot.write(opj(data_dir, f"synth_adj_synth_act_dr6_planck_coadd_{'IQU'[i]}"), p)
    enplot.write(opj(data_dir, f"synth_adj_synth_vs_original_act_dr6_planck_coadd_{'IQU'[i]}"), p_diff)

###############################################################################
############################### Do 2nd method #################################

# perform transform (spin-0 and spin-2). here is where you would try a new
# implementation and compare results!
map_I = sht.synthesis_2d(
    alm=alm_T, spin=0,
    lmax=lmax, geometry=ducc_geometry.name,
    phi0=ducc_info.phi0, ntheta=ducc_geometry.ny, nphi=ducc_geometry.nx,
    nthreads=nthreads
    )

map_QU = sht.synthesis_2d(
    alm=alm_EB, spin=2,
    lmax=lmax, geometry=ducc_geometry.name,
    phi0=ducc_info.phi0, ntheta=ducc_geometry.ny, nphi=ducc_geometry.nx,
    nthreads=nthreads
    )

map_IQU = np.concatenate([map_I, map_QU], axis=0)

# need to extract the data out of the full-sky array now, and 
# flip it back to the original declination and right-ascension ordering.
# finally, reassign original wcs information and write to disk.
map_IQU = map_IQU[..., ducc_geometry.yoff:ducc_geometry.yoff + imap.shape[-2], :]
map_IQU = map_IQU[..., ::-1, ::-1]
map_IQU = enmap.ndmap(map_IQU, masked_imap.wcs)

enmap.write_map(opj(data_dir, 'ducc_map_IQU_synthesis_2d.fits'), map_IQU)

###############################################################################
###############################################################################

# plot to visualize the synethized map and compare to the original map
for i in range(3):
    p = enplot.plot(map_IQU[i], downgrade=32, colorbar=True, ticks=15)
    p_diff = enplot.plot(map_IQU[i] - masked_imap[i], downgrade=32, colorbar=True, ticks=15)

    enplot.write(opj(data_dir, f"synth2d_ana2d_act_dr6_planck_coadd_{'IQU'[i]}"), p)
    enplot.write(opj(data_dir, f"synth2d_ana2d_vs_original_act_dr6_planck_coadd_{'IQU'[i]}"), p_diff)

# plot to visualize the synethized map and compare to the original map
for i in range(3):
    p = enplot.plot(map_IQU[i], downgrade=32, colorbar=True, ticks=15)
    p_diff = enplot.plot(map_IQU[i] - masked_imap[i], downgrade=32, colorbar=True, ticks=15)

    enplot.write(opj(data_dir, f"synth2d_ana2d_act_dr6_planck_coadd_{'IQU'[i]}"), p)
    enplot.write(opj(data_dir, f"synth2d_ana2d_vs_original_act_dr6_planck_coadd_{'IQU'[i]}"), p_diff)