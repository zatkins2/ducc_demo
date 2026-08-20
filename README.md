# ducc_demo
Quick demonstration of [spherical harmonic transforms](https://en.wikipedia.org/wiki/Spherical_harmonics#Spherical_harmonics_expansion) (SHTs) with ducc. The goal is to implement your own package that supplies a module named `sht` with the same functions `adjoint_synthesis`, `analysis_2d`, `synthesis`, and `synthesis_2d` and signatures as are supplied in [`ducc0.sht`](https://mtr.pages.mpcdf.de/ducc/sht.html#sht). 

## Installation and setup
1. Download the repo:
```
git clone git@github.com:zatkins2/ducc_demo.git
```
2. Install the dependencies
```
cd ducc_demo
uv sync
```
3. Select a directory on your system where the inputs and outputs of the test suite will live, `MY_DIR`. Download the inputs (this will take a few minutes)
```
sh download_data.sh MY_DIR
```

## Run
The demo script `ducc_demo.py` gives a minimum working example of how ducc sets up and performs spherical harmonic analysis (map --> alm) and synthesis (alm --> map). Users supply the following command-line arguments to the script:
* `dir` (positional): the directory to read inputs from and into which the outputs are saved (i.e., `MY_DIR`)
* `--shtpack` (optional, default: `ducc0`): the name of a package supplying a module `sht` in which the analogous functions to `ducc0.sht` live. They must have the same signatures for the script to work. Outputs will be prefixed by this package name.
* `--lmax` (optional, default: `10000`): the bandlimit of the spherical harmonic transforms. The default is typical for our applications. Scales with multiples higher than `lmax` (physical scales smaller than about `pi/lmax` radians) are omitted from the transforms.
* `--nthreads` (optional, default: `0`) the number of threads to use. The default tries to infer the number of hardware threads available. This argument may not make sense for a GPU implementation!

Run the script:
```
python ducc_demo.py MY_DIR --shtpack SHTPACK --lmax LMAX --nthreads NTHREADS
```

This will produce data files in `MY_DIR`:
* `SHTPACK_alm_TEB_adjoint_synthesis.npy`: the spherical harmonic coefficients of an ACT DR6 + Planck coadd map, computed using the adjoint of map synthesis.
* `SHTPACK_alm_TEB_analysis_2d.npy`: the same, computed using a dedicated analysis_2d function. Only supported for a handful of pixelization schemes (or, quadratures). The ACT DR6 + Planck map conforms to the "Fejer 1" scheme, see https://arxiv.org/abs/1303.4945.
* `SHTPACK_map_IQU_synthesis.fits`: the map synthesized from the spherical harmonic coefficients. Due to the bandlimiting, this will not be equivalent to the original map.
* `SHTPACK_map_IQU_synthesis_2d.fits`: the same, except limited to the same pixelization schemes.

The script will also produce images to visualize inputs and output maps.

## Notes
The script is heavily (hopefully, helpfully) commented with many details, it's worth reading! 

Most of the script is "boilerplate" meant to set up the call to the spherical harmonic transforms. We use `pixell` for this, which wraps `numpy`, `ducc0`, `astropy.io.fits`, and `astropy.wcs` to support math with oriented, rectangular-pixelized maps on the sphere.

There are calls to `pixell.enmap`, `pixell.curvedsky`, and `ducc0` to extract metadata that is supplied to the `sht` calls. Don't worry about them; all that should be implemented are the `sht` functions themselves! (Of course, feel free to play with modifying these metadata if you like).