from datetime import datetime, timedelta  # noqa: E402
import os  # noqa: E402
import utils  # noqa: E402
import sys  # noqa: E402

os.environ['XRIT_DECOMPRESS_PATH'] = '/gws/smf/j04/nceo_generic/Software/miniconda3/bin/xRITDecompress'

import warnings  # noqa: E402
warnings.filterwarnings('ignore')

if len(sys.argv) < 2:
    proc_dt = datetime.utcnow().replace(microsecond=0, second=0, minute=0)
else:
    dtstr = sys.argv[1]
    try:
        proc_dt = datetime.strptime(dtstr, "%Y%m%d%H%M")
    except ValueError:
        raise ValueError("You must enter a processing date/time in format YYYYMMDDHHMM.")

timedelt = timedelta(hours=1)

# Remote directory into which output data will be saved.
topdir_remote = '/gws/pw/j07/rsg_share/public/nrt/nrt_imagery_geo_global/quick_look_cesium/'

print("Processing:", proc_dt)

# view zenith angle limits. Data not produced above these
vza_thresh_max = 80.

# The composite to produce.
comp = 'natural_color_raw_with_night_ir'

# Limits on reflectance, sometimes these are exceeded due to calibration
# or high solar angle issues. Setting these prevents edge cases from 
# producing odd-looking output.
min_refl = 0
max_refl = 1

# Expected number of files in tile directory
expected_fnum = 2734

# These are for the tiled output used in the VIS tool
zoomlevs = '0-6'
tilesize = 512
processes = 7

# This is the output file on the public folder for use by the VIS tool
output_dir_rsg = f'{topdir_remote}/{proc_dt.strftime("%Y/%m/%d")}/GLOBAL_GEO_{proc_dt.strftime("%Y%m%d%H%M")}_V1_00_FC/'

# Check that the files don't already exist, to save unnecessary reprocessing.
if os.path.exists(output_dir_rsg):
    if utils.totfiles(output_dir_rsg) >= expected_fnum:
        print("Output files already exist on RSGNCEO. Not processing.")
        quit()

# Otherwise, continue
if utils.totfiles(output_dir_rsg) < expected_fnum:
    from multiprocessing import Process, Queue  # noqa: E402
    from osgeo import gdal, gdalconst  # noqa: E402
    import multiprocessing as mp  # noqa: E402
    import numpy as np  # noqa: E402
    import subprocess  # noqa: E402
    import shutil  # noqa: E402

    dirs = utils.DirStruct()

    import satpy
    # Set some satpy options to defaults.
    satpy.config.set(cache_dir=dirs.cache_dir)
    satpy.config.set(tmp_dir=dirs.idir_tmp)
    satpy.config.set(cache_lonlats=True)
    satpy.config.set(cache_sensor_angles=True)
    satpy.config.set(tmp_dir=dirs.tmp_dir)


    proj_str, extent, targ_srs = utils.setup_global_area(res=0.03)

    worp_opts = gdal.WarpOptions(width=targ_srs.width,
                                height=targ_srs.height,
                                outputType=gdal.GDT_Float32,
                                dstSRS=proj_str,
                                dstNodata=-999.,
                                outputBounds=extent,
                                format="vrt",
                                resampleAlg=gdalconst.GRA_Bilinear,
                                multithread=True)

    outf_name = f'{dirs.odir}/GLOBAL_GEO_{proc_dt.strftime("%Y%m%d%H%M")}_V1_00_FC.tif'
    print(outf_name)
    if os.path.exists(outf_name):
        print("Output file exists, skipping image generation for", proc_dt)
    else:
        # Load data
        try:
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []

            # Process SEVIRI PRIME
            p = Process(target=utils.load_seviri, args=(dirs.idir_fds, proc_dt, return_dict, comp, vza_thresh_max, 'fds', worp_opts,))
            p.start()
            jobs.append(p)
            # Process SEVIRI IODC
            p = Process(target=utils.load_seviri, args=(dirs.idir_ioc, proc_dt, return_dict, comp, vza_thresh_max, 'ioc', worp_opts,))
            p.start()
            jobs.append(p)
            # Process ABI GOES-E
            p = Process(target=utils.load_goes, args=(dirs.idir_tmp, proc_dt, return_dict, comp, vza_thresh_max, 'native', 'goes16', worp_opts,))
            p.start()
            jobs.append(p)
            # Process ABI GOES-W
            p = Process(target=utils.load_goes, args=(dirs.idir_tmp, proc_dt, return_dict, comp, vza_thresh_max, 'native', 'goes18', worp_opts,))
            p.start()
            jobs.append(p)
            # Process AHI Himawari-9
            p = Process(target=utils.load_himawari, args=(dirs.idir_tmp, proc_dt, return_dict, comp, vza_thresh_max, 'native', 'B04', worp_opts,))
            p.start()
            jobs.append(p)

            for proc in jobs:
                proc.join()


            # Get the individual datasets
            fds_rgb, fds_vza, gt, sr = return_dict['fds']
            ioc_rgb, ioc_vza, gt, sr = return_dict['ioc']
            g16_rgb, g16_vza, gt, sr = return_dict['goes16']
            g17_rgb, g17_vza, gt, sr = return_dict['goes18']
            hi8_rgb, hi8_vza, gt, sr = return_dict['hi8']

            # Remove NaNs from datasets
            ioc_rgb = utils.remove_baddata_rgb(ioc_rgb, minthresh=min_refl, maxthresh=max_refl)
            fds_rgb = utils.remove_baddata_rgb(fds_rgb, minthresh=min_refl, maxthresh=max_refl)
            g16_rgb = utils.remove_baddata_rgb(g16_rgb, minthresh=min_refl, maxthresh=max_refl)
            g17_rgb = utils.remove_baddata_rgb(g17_rgb, minthresh=min_refl, maxthresh=max_refl)
            hi8_rgb = utils.remove_baddata_rgb(hi8_rgb, minthresh=min_refl, maxthresh=max_refl)

            # Compute which satellite is best to use at a given location
            # First, create a stacked array with all the VZAs
            all_vza = np.dstack((ioc_vza, fds_vza, g16_vza, g17_vza, hi8_vza))
            # Remove bad (fill value or very high) VZA values
            all_vza = np.where(all_vza >= vza_thresh_max, vza_thresh_max, all_vza)
            all_vza = np.where(all_vza < 0.0001, vza_thresh_max, all_vza)
            all_vza = np.where(np.isfinite(all_vza), all_vza, 0)
            all_vza = np.where(np.isnan(all_vza), 0, all_vza)

            # Compute inverse VZA compared to maximum acceptable
            all_vza = vza_thresh_max - all_vza
            all_vza = np.where(all_vza >= vza_thresh_max, 0, all_vza)

            # Compute fraction of each satellite image to be used for a given pixel
            final_vza_frac = all_vza / np.nansum(all_vza, axis=2, keepdims=True)

            # Compute the final output RGB
            out_rgb_arr = (final_vza_frac[:, :, 0].reshape(ioc_rgb.shape[0], ioc_rgb.shape[1], 1) * ioc_rgb +
                        final_vza_frac[:, :, 1].reshape(fds_rgb.shape[0], fds_rgb.shape[1], 1) * fds_rgb +
                        final_vza_frac[:, :, 2].reshape(g16_rgb.shape[0], g16_rgb.shape[1], 1) * g16_rgb +
                        final_vza_frac[:, :, 3].reshape(g17_rgb.shape[0], g17_rgb.shape[1], 1) * g17_rgb +
                        final_vza_frac[:, :, 4].reshape(hi8_rgb.shape[0], hi8_rgb.shape[1], 1) * hi8_rgb)

            out_rgb = np.where(out_rgb_arr > max_refl, max_refl, out_rgb_arr)
            out_rgb = np.where(out_rgb < min_refl, min_refl, out_rgb)

            out_rgb = (255 * out_rgb / max_refl).astype(np.uint8)

            utils.save_img_tiff(out_rgb, outf_name, gt, sr, gdal.GDT_Byte)
        
            utils.rem_old_files(dirs.idir_tmp, proc_dt)
        except:
            utils.rem_old_files(dirs.idir_tmp, proc_dt)
            quit()
    try:
        #import gdal2tiles  # noqa: E402
        #gdal2tiles.generate_tiles(outf_name, output_dir_rsg, zoom=zoomlevs, nb_processes=30, tile_size=tilesize,
        #                          resume=True, webviewer='none')

        # This method uses the external gdal function
        gdal_proc = ['/gws/smf/j04/nceo_generic/Software/miniconda3/bin/python',
                    '-u', '/gws/smf/j04/nceo_generic/Software/miniconda3/bin/gdal2tiles.py', 
                    '--resampling=bilinear',
                    '--zoom=0-7',
                    '--resume',
                    '--tilesize=256',
                    '--webviewer=none',
                    outf_name,
                    output_dir_rsg,
                    '--processes=20']
        subprocess.call(gdal_proc)

        os.remove(outf_name)
    except Exception as e:
        print("Error making tiles for", proc_dt)
        print(e)

else:
    print("Output tiles exist, skipping tile generation for ", proc_dt)


# finally, try moving the files
#if not os.path.exists(output_dir_rsg):
 #   print("Moving to RSGNCEO")
 #   subprocess.call(rsync_opts)
 #   shutil.rmtree(outdir_tiles)
#else:
#    print("Files already exist at RSGNCEO, not moving.")
