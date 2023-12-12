from satpy.modifiers.angles import _get_sensor_angles
from pyresample import create_area_def
from osgeo import gdal
from satpy import Scene
from glob import glob
import numpy as np
import pathlib
import shutil
import copy
import s3fs
import os

import warnings

warnings.filterwarnings('ignore')

bands_abi = ['C02', 'C03', 'C05', 'C11', 'C14', 'C15']
bands_ahi = ['B03', 'B04', 'B05', 'B11', 'B14', 'B15']


class DirStruct:
    def __init__(self,
                 idir_top='/work/scratch-nopw2/proud/global_geo/',
                 tmpdir='/work/scratch-nopw2/proud/tmp/',
                 fdsdir=None,
                 iocdir=None,
                 odir=None,
                 cachedir='/work/scratch-nopw2/proud/tmp/'):
        """Setup directory structure"""
        self.idir_top = idir_top
        if tmpdir is None:
            self.idir_tmp = f'{idir_top}/tmp/'
        else:
            self.idir_tmp = tmpdir

        if fdsdir is None:
            self.idir_fds = '/gws/nopw/j04/nrt_ecmwf_metop/eumetsat/H-000-MSG3/'
        else:
            self.idir_fds = fdsdir

        if iocdir is None:
            self.idir_ioc = '/gws/nopw/j04/nrt_ecmwf_metop/eumetsat/H-000-MSG2/'
        else:
            self.idir_ioc = iocdir

        if odir is None:
            self.odir = f'{idir_top}/out/'
        else:
            self.odir = odir

        if cachedir is None:
            self.cache_dir = f'{idir_top}/cache/'
        else:
            self.cache_dir = cachedir
        
        pathlib.Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.odir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.idir_tmp).mkdir(parents=True, exist_ok=True)
            

def totfiles(idir):
    total = 0
    for root, dirs, files in os.walk(idir):
        total += len(files)
    return total


def rem_old_files(datadir, proc_dt):
    """Delete unneeded files from processing."""
    ahi_f = glob(f'{datadir}/*{proc_dt.strftime("%Y%m%d_%H%M")}*')
    for fil in ahi_f:
        os.remove(fil)
    goes_f = glob(f'{datadir}/*_s{proc_dt.strftime("%Y%j%H%M")}*')
    for fil in goes_f:
        os.remove(fil)


def get_ahi_band(infname):
    pos = infname.find('_FLDK_')
    return infname[pos - 3:pos]


def get_abi_band(infname):
    pos = infname.find('_G16_s2')
    if pos < 0:
        pos = infname.find('_G17_s2')
        if pos < 0:
            pos = infname.find('_G18_s2')

    return infname[pos - 3:pos]


def dl_himawari(idir, dater):
    output_files_h08 = []

    himi08_s3str = f's3://noaa-himawari9/AHI-L1b-FLDK/{dater.strftime("%Y/%m/%d/%H%M/")}'
    himi08_dtstr = dater.strftime("%Y%m%d_%H%M")
    
    s3 = s3fs.S3FileSystem(anon=True)
    the_files_h08 = s3.ls(himi08_s3str)

    print('Retrieving Himawari-9')
    for fname in the_files_h08:
        bname = get_ahi_band(fname)
        if bname not in bands_ahi:
            continue
        if himi08_dtstr not in fname:
            continue
            
        pos = fname.rfind('/')
        fname2 = fname[pos + 1:]
        outf = f'{idir}/{fname2}'
        outf2 = outf[:-4]
        if not os.path.exists(outf) and not os.path.exists(outf2):
            s3.get(fname, outf)
            output_files_h08.append(outf)
        else:
            if os.path.exists(outf):
                output_files_h08.append(outf)
            else:
                output_files_h08.append(outf2)

    return output_files_h08
    

def dl_goes(idir, dater, sat):
    output_files = []

    goes_s3str = f's3://noaa-{sat}/ABI-L1b-RadF/{dater.strftime("%Y/%j/%H/")}'
    goes_dtstr = dater.strftime("%Y%j%H%M")

    s3 = s3fs.S3FileSystem(anon=True)
    the_files_goes = s3.ls(goes_s3str)

    print('Retrieving', sat)
    for fname in the_files_goes:
        bname = get_abi_band(fname)
        if bname not in bands_abi:
            continue
        if goes_dtstr not in fname:
            continue
            
        pos = fname.rfind('/')
        fname2 = fname[pos + 1:]
        outf = f'{idir}/{fname2}'
        if not os.path.exists(outf):
            s3.get(fname, outf)
        output_files.append(outf)

    return output_files


def norm_output(in_rgb, max_refl, min_refl):
    """
    Args:
        in_rgb: Numpy array, the global RGB to normalise
        max_refl: Maximum acceptable reflectance
        min_refl: Minimum acceptable reflectance
    Returns:
        out_rgb: Numpy array, the normalised RGB
    """
    out_rgb = copy.deepcopy(in_rgb)
    out_rgb = np.where(out_rgb > max_refl, max_refl, out_rgb)
    out_rgb = np.where(out_rgb < min_refl, min_refl, out_rgb)
    return (255 * out_rgb / max_refl).astype(np.uint8)


def create_vza_frac(ioc, fds, g16, g17, hi8, vza_thresh_max):
    """
    Arguments:
        ioc: Numpy array, Indian Ocean SEV view zenith angle.
        fds: Numpy array, Prime SEV view zenith angle.
        g16: Numpy array, GOES-E ABI view zenith angle.
        g17: Numpy array, GOES-W ABI view zenith angle.
        hi8: Numpy array, Himawari-8 AHI view zenith angle.
        vza_thresh_max: Float, the maximum acceptable VZA in degrees.
    Returns:
        -   vza_frac: Numpy array giving the fraction of each satellite scene to use in output image.
    """

    # Compute which satellite is best to use at a given location
    # First, create a stacked array with all the VZAs
    all_vza = np.dstack((ioc, fds, g16, g17, hi8))
    # Remove bad (fill value or very high) VZA values
    all_vza = np.where(all_vza >= vza_thresh_max, vza_thresh_max, all_vza)
    all_vza = np.where(all_vza < 0.0001, vza_thresh_max, all_vza)
    all_vza = np.where(np.isfinite(all_vza), all_vza, 0)
    all_vza = np.where(np.isnan(all_vza), 0, all_vza)

    # Compute inverse VZA compared to maximum acceptable
    all_vza = vza_thresh_max - all_vza
    all_vza = np.where(all_vza >= vza_thresh_max, 0, all_vza)

    # Compute fraction of each satellite image to be used for a given pixel
    return all_vza / np.nansum(all_vza, axis=2, keepdims=True)


def setup_global_area(res=0.05):
    """Create an area extent onto which datasets are reprojected.
    Arguments:
        - res: Float, the output image resolution in degrees.
    Returns:
        - proj_str: String, the PROJ4 string describing the projection.
        - extent: The extent of the output image in degrees.
        - targ_srs: The spatial reference source of the output.
    """
    area_ext = (-180, -65, 180, 65)
    res = (res, res)
    targ_srs = create_area_def("source_area", "EPSG:4326", area_extent=area_ext, resolution=res)
    proj_str = targ_srs.proj_str
    extent = (targ_srs.area_extent[0],
              targ_srs.area_extent[1],
              targ_srs.area_extent[2],
              targ_srs.area_extent[3])

    return proj_str, extent, targ_srs


def _retr_satgeom(area):
    """Retrieve satellite position from Scene if not supplied by user."""
    if area.proj_dict['proj'] == 'geos':
        sat_lat = 0.0
        sat_lon = area.proj_dict['lon_0']
        sat_alt = area.proj_dict['h'] / 1000.
    else:
        raise TypeError('Automatic satellite position calculation requires geostationary data.')

    return [sat_lon, sat_lat, sat_alt]


def _make_common_ds(scn, composite, vza_thresh=75.):
    """Common functions for all satellite types, computes VZA and makes GDAL DS.
    Arguments:
        - scn: Scene, the datasets themselves.
        - composte: String, the composite/dataset name to load.
        - vza_thresh: Float, the threshold for chopping VZA.
    Returns:
        - out_ds: Gdal dataset, containing the composite and vza.
    """
    scn_sr = scn[composite].attrs['area'].crs_wkt
    scn_res = scn[composite].attrs['area'].resolution
    scn_ext = scn[composite].attrs['area'].area_extent
    scn_rot = 0 # scn[composite].attrs['area'].rotation <---previously used this but deprecated now
    scn_width = scn[composite].attrs['area'].width
    scn_height = scn[composite].attrs['area'].height
    scn_gt = (scn_ext[0], scn_res[0], scn_rot, scn_ext[2], scn_rot, -scn_res[1])

    data_in = scn[composite].values
    scn_vaa, scn_vza = _get_sensor_angles(scn[composite])
    scn_vza = np.array(scn_vza)

    scn_vza = np.where(scn_vza < vza_thresh, scn_vza, np.nan)

    driver = gdal.GetDriverByName('MEM')
    ds = driver.Create("myraster", scn_width, scn_height, 4, gdal.GDT_Float32)
    ds.SetProjection(scn_sr)
    ds.SetGeoTransform(scn_gt)
    ds.GetRasterBand(1).WriteArray(data_in[0, :, :])
    ds.GetRasterBand(2).WriteArray(data_in[1, :, :])
    ds.GetRasterBand(3).WriteArray(data_in[2, :, :])
    ds.GetRasterBand(4).WriteArray(scn_vza)

    return ds


def remove_baddata_rgb(indata, minthresh=0, maxthresh=120):
    """Remove bad data from an image array.
    Arguments:
        - indata: Numpy array containing the rgb data.
        - minthresh: Float, minimum acceptable pixel value
        - maxthresh: Float, maximum acceptable pixel value
    Returns:
        - indata: The modified input data
    """

    indata = np.clip(indata, minthresh, minthresh)
    indata = np.where(np.isfinite(indata) != True, 0, indata)

    return indata


def save_img_tiff(data, fname, gt, sr, dtype=gdal.GDT_Float32):
    """Save an image to disk."""
    shaper = data.shape
    n_bands = 1
    if len(shaper) > 2:
        n_bands = shaper[2]
    output_raster = gdal.GetDriverByName('GTiff').Create(fname, shaper[1], shaper[0], n_bands, dtype,
                                                         options=['COMPRESS=LZW'])
    output_raster.SetGeoTransform(gt)
    output_raster.SetProjection(sr)
    if n_bands == 1:
        output_raster.GetRasterBand(1).WriteArray(data[:, :])
    else:
        for i in range(0, n_bands):
            output_raster.GetRasterBand(i + 1).WriteArray(data[:, :, i])
    output_raster.FlushCache()
    del output_raster


def load_seviri(indir, dater, thequeue, composite, vza_thresh, mission, opts):
    """Load a MSG/SEVIRI image.
    Arguments:
        - indir: String, the directory containing SEVIRI HRIT data.
        - dater: Datetime, the time to process.
        - thequeue: dict, stores results
        - composite: String, the composite/dataset name to load.
        - vza_thresh: Float, the threshold for chopping VZA.
        - mission: String for mission name
        - opts: WarpOptions for GDAL
    Returns:
        - scn_ds: Gdal dataset containing scene composite and view zenith angle.
    """
    dirstr = dater.strftime("%Y/%m/%d/")
    curfiles = glob(f'{indir}/{dirstr}/*{dater.strftime("%Y%m%d%H%M")}*')
    if len(curfiles) < 50:
        raise ValueError("Not enough SEVIRI data for processing.")

    scn = Scene(curfiles, reader='seviri_l1b_hrit')
    scn.load([composite], upper_right_corner='NE')
    
    cur_ds = _make_common_ds(scn, composite, vza_thresh)
    
    rgb, vza, gt, sr = resample_img(cur_ds, opts)
   
    thequeue[mission] = (rgb, vza, gt, sr)

    return 


def load_goes(indir, dater, thequeue, composite, vza_thresh, resampler, sat, opts):
    """Load a GOES/ABI image.
    Arguments:
        - indir: String, the directory containing ABI netCDF data.
        - dater: Datetime, the time to process.
        - thequeue: dict, stores results
        - composite: String, the composite/dataset name to load.
        - vza_thresh: Float, the threshold for chopping VZA.
        - resampler: String, the resampling method for satpy to use.
        - sat: String, which satellite to process. 'goes16' or 'goes17' or 'goes18'
        - opts: WarpOptions for GDAL
    Returns:
        - scn_ds: Gdal dataset containing scene composite and view zenith angle.
    """

    curfiles = dl_goes(indir, dater, sat)

    if len(curfiles) < 4:
        raise ValueError("Not enough GOES ABI data for processing.")

    scn = Scene(curfiles, reader='abi_l1b')
    scn.load([composite], generate=False)
    scn = scn.resample(scn.coarsest_area(), resampler=resampler)
    
    cur_ds = _make_common_ds(scn, composite, vza_thresh)
    
    rgb, vza, gt, sr = resample_img(cur_ds, opts)
   
    thequeue[sat] = (rgb, vza, gt, sr)

    return


def load_himawari(indir,
                  dater,
                  thequeue,
                  composite,
                  vza_thresh,
                  resampler,
                  ref_band,
                  opts):
    """Load a GOES/ABI image.
    Arguments:
        - indir: String, the directory containing AHI HSD data.
        - dater: Datetime, the time to process.
        - thequeue: dict, stores results
        - composite: String, the composite/dataset name to load.
        - vza_thresh: Float, the threshold for chopping VZA.
        - resampler: String, the resampling method for satpy to use.
        - ref_band: String, the band to load orbital parameters from.
        - opts: WarpOptions for GDAL
    Returns:
        - scn_ds: Gdal dataset containing scene composite and view zenith angle.
    """

    curfiles = dl_himawari(indir, dater)

    if len(curfiles) < 4:
        print(curfiles)
        raise ValueError("Not enough Himawari HSD data for processing.")

    scn = Scene(curfiles, reader='ahi_hsd')
    scn.load([composite, ref_band], generate=False)
    scn = scn.resample(scn.coarsest_area(), resampler=resampler)
    scn[composite].attrs['orbital_parameters'] = scn[ref_band].attrs['orbital_parameters']

    cur_ds = _make_common_ds(scn, composite, vza_thresh)

    rgb, vza, gt, sr = resample_img(cur_ds, opts)

    thequeue['hi8'] = (rgb, vza, gt, sr)
    return


def resample_img(in_ds, opts, fname=''):
    """Use GDAL to resample input data onto another grid.
    Arguments:
        - in_ds: Gdal dataset, the input dataset that will be resampled.
        - opts: WarpOptions, the options for the warp.
        - fname: String, filename to save results into. Leave blank in most cases.
                 If not blank, ops must have format set to something other than 'vrt'.
    Returns:
        - out_rgb: Resampled RGB composite data
        - out_vza: Resampled view zenith angles.
    """
    ods = gdal.Warp(destNameOrDestDS=fname, srcDSOrSrcDSTab=in_ds, options=opts)
    gt = ods.GetGeoTransform()
    sr = ods.GetProjection()

    b1 = ods.GetRasterBand(1).ReadAsArray()
    b2 = ods.GetRasterBand(2).ReadAsArray()
    b3 = ods.GetRasterBand(3).ReadAsArray()

    out_vza = ods.GetRasterBand(4).ReadAsArray()

    out_rgb = np.dstack((b1, b2, b3))

    del in_ds
    del ods

    return out_rgb, out_vza, gt, sr


def adjust_gamma(image, gamma=1.0):
    """Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values."""
    invgamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invgamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

'''
def write_cv2_img(fname, data):
    """Write a numpy array to disk using CV2."""
    import cv2
    hsv = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = np.log(mid * 255) / np.log(mean)
    adjusted = adjust_gamma(data, gamma=gamma)
    cv2.imwrite(fname, cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR))
'''
