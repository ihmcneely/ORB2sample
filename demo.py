import hurdat
import GOES
import ORB
import matplotlib.pyplot as plt
import numpy as np
import copy as cp

# In order to download from the merged IR database, you will need to create an
# account on https://disc.gsfc.nasa.gov/ and follow the instructions at
# https://disc.gsfc.nasa.gov/data-access
user = 'imcneely'
password = 'opendapWD-40_ORB'

# Download hurdat database
data = hurdat.Hurdat()

# data.storms is a pandas frame with the timestamp, name, id, location, and
# intensity of all HURDAT TCs in the NAL, ENP, and CP. We have a helper
# function to linearly interpolate down to a hourly best track.
# For Edouard [2014]:
storm = data.get_hourly_position('AL062014')

# We then add the radius of the stamp + a buffer for smoothing, then download
# from MERGEIR. Only 6 hours of stamps are downloaded here.
stormList = cp.copy(storm)
stormList['RADIUS'] = 1000
stamps = GOES.Stamp.from_list_MERGEIR(stormList[102:109],
                                      user,
                                      password)

# We can plot using the built-ins for xarray.
plt.figure()
stamps.data.isel(time=0).temperature.plot()

# Create an ORB object from a sequence of stamps and plot the first stamp.
edouard = ORB.ORB(stamps)
plt.figure()
edouard.plot(time=0)

# Compute a useful layer.
edouard.add_radius()

# Computing a radial profile for temperature.
RAD = edouard.compute_radial_profile(nsector=1)
RAD = RAD.rename({'profile': 'radial_profile'})
plt.figure()
RAD.isel(time=0).radial_profile.plot()

# Prepare radial profiles for saving to csv
RAD_csvprepped = RAD.drop('sector').radial_profile.to_dataframe().unstack(level=0)