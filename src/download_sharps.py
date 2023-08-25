from __future__ import absolute_import, division, print_function
import os
import drms
import pandas as pd
import sys
import time

# Print the doc string of this example.
print(__doc__)


# If you don't want to enter your email address during program execution, you
# can set this variable to the email address you have registered for JSOC data
# exports. If you have not registered your email yet, you can do this on the
# JSOC website at: http://jsoc.stanford.edu/ajax/register_email.html
email = 'kiera.vds@gmail.com'
save_dir = '/d0/kvandersande/sharps/'
# Use 'as-is' instead of 'fits', if record keywords are not needed in the
# FITS header. This greatly reduces the server load!
export_protocol = 'fits'
#export_protocol = 'as-is'

# Series, harpnum, timespan and segment selection
series = 'hmi.sharp_cea_720s'
#series = 'aia.lev1_uv_24s'
#series = 'hmi.M_720s'

#series = 'hmi.Ic_720s'
harpnum = 4864
#tsel = '2017.09.07_10:00:00_TAI/12m@12m'
tsel = '2010.01.01_07:48:00_TAI/7000d@8h'
#tsel = '2010.06.01_00:00:00_TAI/2d@1h'
segments = ['Bp', 'Br', 'Bt','magnetogram']
#segments = ['Br']
print("Starting the download ...")
#segments = ['magnetogram']
#segments = ['image']
harpnums = pd.read_csv('sharps_with_noaa_ars.csv')['HARPNUM']
harpnums = harpnums[(harpnums>=int(sys.argv[1]))&(harpnums<=int(sys.argv[2]))]

for harpnum in harpnums:
    print("Downloading sharp", harpnum)


    # Download directory
    out_dir = os.path.join(save_dir, 'sharp_%d' % harpnum)

    # Create download directory if it does not exist yet.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create DRMS client, use debug=True to see the query URLs.
    c = drms.Client(verbose=True)

    # Data export query string
    qstr = '%s[%d][%s]{%s}' % (series, harpnum, tsel, ','.join(segments))
    #qstr = '%s[%d][1600]{%s}' % (series, tsel, ','.join(segments))
    print('Data export query:\n  %s\n' % qstr)

    # Submit export request using the 'fits' protocol
    t0 = time.time()

    print('Submitting export request...')
    export_request = c.export(qstr, method='url', protocol=export_protocol, email=email)

    # loop to wait for query to go through
    while True:
        try: 
            if export_request.status == 2 and export_request.id != None:
                export_request.wait(timeout=20*60)
                break
            time.sleep(10)
        except:
            time.sleep(10)
        if time.time()-t0 > 20*60:
            print('Failed to export query after 20 min...')
            break

    # check status
    status = export_request.status
    print("     client.export.status = ",status)
    if status != 0:
        print("*********** DRMS error for SHARP ", harpnum)
        break
    print("         Export request took ", time.time() - t0, ' seconds')

    # Download the files
    t1 = time.time()

    print('\nRequest URL: %s' % export_request.request_url)
    if '%s' % export_request.request_url != 'None':
        print('%d file(s) available for download.\n' % len(export_request.urls))
    print("     Running the download command...")

    export = export_request.download(out_dir, verbose=False)
    print("          Download took: ", time.time() - t1, "seconds")

