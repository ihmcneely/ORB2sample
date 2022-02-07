import GOES
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from scipy.stats import gaussian_kde
import numpy as np
from scipy import signal
import pandas as pd
import xarray as xa
import statistics as stat
import copy
import warnings
from sklearn.decomposition import PCA
import re
from scipy import ndimage


def wrap_angle_pm_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def deviation_wrap(angle):
    t1 = angle < 0
    angle = np.multiply((360 + angle), t1) + np.multiply(angle, (1 - t1))
    t1 = angle < 90
    t2 = angle >= 270
    angle = (np.multiply(angle, t1) +
        np.multiply((angle - 180), (1 - (t1 + t2))) +
        np.multiply((angle - 360), t2))
    return angle


def ceil_to_base(x, base=5):
    return base * np.ceil(x / base)


def angular_mean(values, axis=None):
    x = np.cos(values)
    y = np.sin(values)
    return np.arctan2(np.nanmean(y, axis=axis), np.nanmean(x, axis=axis))


def angular_var(values, axis=None):
    x = np.cos(values)
    y = np.sin(values)
    return 1 - np.sqrt(np.add(
        np.square(np.nanmean(x, axis=axis)), np.square(np.nanmean(y, axis=axis))
    ))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def minimum_wind(id, hurdat, threshold):
    aboveThreshold = hurdat.storms[hurdat.storms.ID == id].WIND >= threshold
    start = hurdat.storms[
        hurdat.storms.ID == id
        ].DATETIME[aboveThreshold].min()
    end = hurdat.storms[
        hurdat.storms.ID == id
        ].DATETIME[aboveThreshold].max()
    return (slice(start, end))


def blur(img, ksize=(0, 0), sigmaX=None, sigmaY=None):
    # replace nan with 0 in imgA, imgB, set imgB to 1 (for weight correction)
    imgA = img.copy()
    imgA[np.isnan(imgA)] = 0
    imgB = 0 * img.copy() + 1
    imgB[np.isnan(imgB)] = 0

    blurA = cv2.GaussianBlur(
        src=imgA, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY
    )
    blurB = cv2.GaussianBlur(
        src=imgB, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY
    )
    return blurA / blurB


# DFS code drawn from https://www.geeksforgeeks.org/find-number-of-islands/
# Modified to use iterators following the advice of:
# https://stackoverflow.com/questions/28660685/recursion-depth-issue-using-python-with-dfs-algorithm
class Graph:
    def __init__(self, row, col, g):
        self.ROW = row
        self.COL = col
        self.graph = g
        self.sets = np.zeros(g.shape)

    # A function to check if a given cell
    # (row, col) can be included in DFS
    def isSafe(self, i, j, visited):
        # row number is in range, column number
        # is in range and value is non-zero
        # and not yet visited
        return (i >= 0 and i < self.ROW and
                j >= 0 and j < self.COL and
                not visited[i][j] and self.graph[i, j] >= 0)

    # A utility function to do DFS for a 2D
    # boolean matrix. It only considers
    # the 8 neighbours as adjacent vertices.
    # Mark valid pixels along the way
    def DFS(self, i, j, visited):
        # These arrays are used to get row and
        # column numbers of 8 neighbours
        # of a given cell
        rowNbr = [-1, -1, -1, 0, 0, 1, 1, 1]
        colNbr = [-1, 0, 1, -1, 1, -1, 0, 1]

        # Mark first cell as visited
        visited[i][j] = True
        self.sets[i, j] += 1

        istack = [iter([i + x for x in rowNbr])]
        jstack = [iter([j + x for x in colNbr])]
        while istack:
            try:
                ii = next(istack[-1])
                jj = next(jstack[-1])
                if self.isSafe(ii, jj, visited):
                    visited[ii][jj] = True
                    self.sets[ii, jj] += 1
                    istack.append(iter([ii + x for x in rowNbr]))
                    jstack.append(iter([jj + x for x in colNbr]))
            except StopIteration:
                istack.pop()
                jstack.pop()

    # The main function that returns
    # count of islands in a given boolean
    # 2D matrix
    def mark_valid_islands(self, radThreshold):
        # Make a bool array to mark visited cells.
        # Initially all cells are unvisited
        visited = [[0 for j in range(self.COL)] for i in range(self.ROW)]

        # Initialize count as 0 and traverse
        # through the all cells of
        # given matrix
        for i in range(self.ROW):
            for j in range(self.COL):
                # only start islands from within a given radius
                if self.graph[i][j] > radThreshold:
                    continue
                # If a cell with value 1 is not visited yet,
                # then new valid island found
                if not visited[i][j] and self.graph[i][j] >= 0:
                    # Visit all cells in this island and mark
                    self.DFS(i, j, visited)

        return self.sets

class ORB:
    def __init__(self, GOES):
        self.data = copy.copy(GOES.data)
        self.data = self.data.assign({
            'centered_lat': (['time'], [0.0] * len(self.data.time.values)),
            'centered_lon': (['time'], [0.0] * len(self.data.time.values))
        })

    ### ORB related functions
    def add_radius(self):
        "Adds variable with great circle distance to center of stamp."
        self.data = self.data.assign(
            {'radius': GOES.distance_from_coord(
                self.data.LATCENTER + self.data.centered_lat * 0.04,
                self.data.coords['lat'] + self.data.LATSHIFT,
                self.data.LONCENTER + self.data.centered_lon * 0.04,
                self.data.coords['lon'] + self.data.LONSHIFT
            )}
        )

    def add_diff(self, variable='temperature', smooth=False, tau=1,
                 filter='exponential'):
        tmp = self.data[variable].transpose('time', 'lat', 'lon').values
        for tt in range(tmp.shape[0]-1):
            if np.sum(np.isnan(tmp[tt+1,:,:]))/(tmp.shape[1]*tmp.shape[2]) > .5:
                tmp[tt+1, :, :] = tmp[tt, :, :]
            else:
                tmp[tt+1, np.isnan(tmp[tt+1, :, :])] = tmp[tt, np.isnan(tmp[tt+1, :, :])]
        if smooth:
            if filter == 'exponential':
                # tau gives half life in hours
                weights = np.exp(
                    np.linspace(0, 2*tau*5, 2*tau*5)*np.log(.5)/(2*tau)
                )
            elif filter == 'half-gaussian':
                # tau gives sd in hours
                weights = np.exp(
                    -(np.linspace(0, 2*tau*2, 2*tau*2) / (2*tau)) ** 2
                )

            weights = weights / np.sum(weights)
            tmp = ndimage.convolve(tmp,
                                   np.expand_dims(weights, axis=(1, 2)),
                                   mode='nearest',
                                   origin=(-len(weights)//2, 0, 0))
        tmp = np.diff(tmp, prepend=0, axis=0)
        tmp[0, :, :] = tmp[1, :, :]
        self.data = self.data.assign(
            {variable+'_diff': (
                ['time', 'lat', 'lon'],
                np.diff(tmp, prepend=0, axis=0)
            )}
        )

    def find_eye(self, threshold=30, threshold2=0, closureThreshold=.75,
                 outerRad=150):
        def quick_polar(img, x=0, y=0):
            x0 = np.floor(img.shape[0] / 2) + x
            y0 = np.floor(img.shape[1] / 2) - y

            r = [r for r in range(np.floor(np.min(img.shape) / 2).__int__())]
            theta = [[t * np.pi / 180] * r.__len__() for t in range(360)]
            r = r * 360
            theta = [inner
                     for outer in theta
                     for inner in outer
                     ]

            x1 = np.round(x0 + r * np.cos(theta)).astype('int')
            x1 = [clamp(x, 0, img.shape[0] - 1) for x in x1]
            y1 = np.round(y0 + r * np.sin(theta)).astype('int')
            y1 = [clamp(y, 0, img.shape[1] - 1) for y in y1]

            return img[x1, y1].reshape((360, int(r.__len__() / 360))).transpose()

        def get_eye_closure(cLat, cLon, img):
            # Compute radial profile, check for eye
            polar = quick_polar(img, cLat, cLon)
            radial = np.nanmean(polar, axis=1)  # approximated radial profile
            minIdx = np.nanargmin(radial)
            if minIdx > 0:
                maxEye = np.nanmax(radial[:minIdx])
            else:
                maxEye = -9999
            fullEye = maxEye - radial[minIdx] - threshold > 0  # Eye/eyewall difference sufficiently large
            fullEye = fullEye and (maxEye > threshold2)  # Eye temperature sufficiently high
            if fullEye:
                roundMinIdx = np.nanargmin(polar, axis=0)
                roundMin = np.array(
                    [polar[roundMinIdx[xx], xx]
                     for xx in range(polar.shape[1])]
                )
                roundMaxEye = [0]*polar.shape[1]
                for xx in range(polar.shape[1]):
                    if roundMinIdx[xx] > 0:
                        roundMaxEye[xx] = np.nanmax(
                            polar[:roundMinIdx[xx], xx]
                        )
                    else:
                        roundMaxEye[xx] = -9999

                return np.mean((roundMaxEye - roundMin - threshold) > 0)
            else:
                return 0

        latShifts = np.zeros(len(self.data.time.values))
        lonShifts = np.zeros(len(self.data.time.values))
        closures = np.zeros(len(self.data.time.values))

        for ii in range(len(self.data.time.values)):
            time = self.data.time.values[ii]

            bounds = GOES.coord_from_distance(
                self.data.LATSHIFT.sel(time=time).values, 0, outerRad
            )
            bounds['latHi'] -= np.max(self.data.LATSHIFT.values[ii])
            bounds['latLo'] -= np.max(self.data.LATSHIFT.values[ii])
            lats = [x for x in self.data.lat.values if bounds['latLo'] <= x <= bounds['latHi']]
            lons = [x for x in self.data.lon.values if bounds['lonLo'] <= x <= bounds['lonHi']]

            local = self.data.sel(lat=slice(min(lats), max(lats)),
                                  lon=slice(min(lons), max(lons)),
                                  time=time).temperature.values

            if np.mean(np.isnan(local)) > .1:  # Skip largely missing images
                continue
            blurLocal = blur(local, ksize=(13, 13),
                             sigmaX=50*.04, sigmaY=50*.04)
            center = np.where(blurLocal == np.nanmax(blurLocal))
            center = (
                clamp(
                    center[0][0] - np.where(np.abs(lats) < 1e-6)[0][0], -20, 20
                ),
                clamp(
                    center[1][0] - np.where(np.abs(lons) < 1e-6)[0][0], -20, 20
                )
            )

            # Check if eye exists and report closure fraction.
            closures[ii] = get_eye_closure(center[0], center[1], local)
            if closures[ii] > closureThreshold:
                latShifts[ii] = center[0]
                lonShifts[ii] = center[1]

        return ({
            'closure_fraction': closures,
            'centeredLat': latShifts,
            'centeredLon': lonShifts
        })

    def center_on_eye(self, threshold=30, threshold2=0, closureThreshold=.75,
                      outerRad=150):
        eyeInfo = self.find_eye(threshold=threshold,
                                threshold2=threshold2,
                                closureThreshold=closureThreshold,
                                outerRad=outerRad)

        self.data.centered_lat.values += eyeInfo['centeredLat']
        self.data.centered_lon.values += eyeInfo['centeredLon']

    def add_blur(self, blurRadius=50, resolution=0.04, kernel='other'):
        """
           This function smooths the brightness temperatures within ORB.data

           Parameters:
           blurRadius (float): radius of blurring in km
           """
        # Lat-lon ratio will only vary with lat
        self.data = self.data.assign(
            {'latlon_ratio': (['lat', 'time'],
                              GOES.distance_from_coord(
                                  self.data.lat + self.data.LATSHIFT,
                                  self.data.lat + self.data.LATSHIFT,
                                  self.data.lon[0] + self.data.LONSHIFT,
                                  self.data.lon[0] + self.data.LONSHIFT + resolution
                              ) / GOES.distance_from_coord(
                                  self.data.lat + self.data.LATSHIFT,
                                  self.data.lat + self.data.LATSHIFT + resolution,
                                  self.data.lon[0] + self.data.LONSHIFT,
                                  self.data.lon[0] + self.data.LONSHIFT
                              )
                              )}
        )

        # Create smoothed layer. Currently uses center of image to compute a
        # constant stride for full image. Should vary latStride with lon.
        kernBox = GOES.coord_from_distance(
            self.data['lat'][round(len(self.data['lat']) / 2)],
            self.data['lon'][0],
            blurRadius / 2
        )
        kernStride = {
            'latStride': (kernBox['latHi'] - kernBox['latLo']) / resolution,
            'lonStride': (kernBox['lonHi'] - kernBox['lonLo']) / resolution
        }

        smooth = np.zeros(self.data.temperature.shape)
        for ii in range(smooth.shape[0]):
            smooth[ii, :, :] = blur(
                self.data.temperature.isel(time=ii).values,
                sigmaX=kernStride['lonStride'],
                sigmaY=kernStride['latStride']
            )

        self.data = self.data.assign(
            {'smoothed_temperature': (['time', 'lat', 'lon'], smooth)}
        )

    def add_gradient(self, blurRadius=50, resolution=0.04, kernel='other'):
        """
           Compute the image gradient and add corresponding layers to the stamp.

           This function smooths the brightness temperatures within ORB.data
           and computes the gradient in the latitudinal and longitudinal
           directions. The resulting gradients are converted to polar
           coordinates and all 5 layers are stored.

           Parameters:
           blurRadius (float): radius of blurring in km
           """
        # Lat-lon ratio will only vary with lat
        self.data = self.data.assign(
            {'latlon_ratio': (['lat', 'time'],
                              GOES.distance_from_coord(
                                  self.data.lat + self.data.LATSHIFT,
                                  self.data.lat + self.data.LATSHIFT,
                                  self.data.lon[0] + self.data.LONSHIFT,
                                  self.data.lon[0] + self.data.LONSHIFT + resolution
                              ) / GOES.distance_from_coord(
                                  self.data.lat + self.data.LATSHIFT,
                                  self.data.lat + self.data.LATSHIFT + resolution,
                                  self.data.lon[0] + self.data.LONSHIFT,
                                  self.data.lon[0] + self.data.LONSHIFT
                              )
                              )}
        )

        # Create smoothed layer. Currently uses center of image to compute a
        # constant stride for full image. Should vary latStride with lon.
        kernBox = GOES.coord_from_distance(
            self.data['lat'][round(len(self.data['lat']) / 2)],
            self.data['lon'][0],
            blurRadius / 2
        )
        kernStride = {
            'latStride': (kernBox['latHi'] - kernBox['latLo']) / resolution,
            'lonStride': (kernBox['lonHi'] - kernBox['lonLo']) / resolution
        }

        smooth = np.zeros(self.data.temperature.shape)
        for ii in range(smooth.shape[0]):
            smooth[ii, :, :] = blur(
                self.data.temperature.isel(time=ii).values,
                sigmaX=kernStride['lonStride'],
                sigmaY=kernStride['latStride']
            )

        self.data = self.data.assign(
            {'smoothed_temperature': (['time', 'lat', 'lon'], smooth)}
        )

        # Compute the lat/lon gradients
        def get_grad(img, x=0, y=1, option='numpy'):
            if option == 'sobel':
                out = cv2.Sobel(img, cv2.CV_64F, x, y, 1)
            elif option == 'numpy':
                if (x == 0) & (y == 1):
                    out = np.gradient(img, axis=0)
                elif (x == 1) & (y == 0):
                    out = np.gradient(img, axis=1)
                else:
                    out = np.NaN
            else:
                r = np.array([-1, -1, 0, -1, -1])
                h = -np.vstack([r, r, np.zeros((5)), -r, -r])
                if (x == 0) & (y == 1):
                    out = signal.convolve2d(img, h, mode='same',
                                            boundary='fill')
                elif (x == 1) & (y == 0):
                    out = signal.convolve2d(img, h.transpose(), mode='same',
                                            boundary='fill')
                else:
                    out = np.NaN
            return out

        latgrad = np.zeros(self.data.temperature.shape)
        longrad = np.zeros(self.data.temperature.shape)
        for ii in range(smooth.shape[0]):
            latgrad[ii, :, :] = get_grad(
                self.data.smoothed_temperature.isel(time=ii).values, x=0, y=1,
                option=kernel
            ) * self.data.latlon_ratio.isel(time=ii).values[:, np.newaxis]
            longrad[ii, :, :] = get_grad(
                self.data.smoothed_temperature.isel(time=ii).values, x=1, y=0,
                option=kernel
            )

        self.data = self.data.assign(
            {'lat_gradient': (['time', 'lat', 'lon'], latgrad),
             'lon_gradient': (['time', 'lat', 'lon'], longrad)
             }
        )

        # compute the magnitude and direction
        def mag(imgLat, imgLon):
            return np.sqrt(np.square(imgLat) + np.square(imgLon))

        magimg = np.zeros(self.data.temperature.shape)
        ang = np.zeros(self.data.temperature.shape)
        for ii in range(smooth.shape[0]):
            magimg[ii, :, :] = mag(
                self.data.lat_gradient.isel(time=ii).values,
                self.data.lon_gradient.isel(time=ii).values
            )
            ang[ii, :, :] = np.arctan2(
                self.data.lat_gradient.isel(time=ii).values,
                self.data.lon_gradient.isel(time=ii).values
            )

        self.data = self.data.assign(
            {
                'mag_gradient': (['time', 'lat', 'lon'], magimg),
                'angle_gradient': (['time', 'lat', 'lon'], ang)
            }
        )
        self.kernStride = kernStride
        self.kernRadius = blurRadius

    def add_divergence(self):
        self.data = self.data.assign(
            {'divergence': (['time', 'lat', 'lon'],
                np.gradient(self.data.lon_gradient, axis=2)
              + np.gradient(self.data.lat_gradient, axis=1)
            )}
        )

    def add_curl(self):
        self.data = self.data.assign(
            {'curl': (['time', 'lat', 'lon'],
                np.gradient(self.data.lat_gradient, axis=2) -
                np.gradient(self.data.lon_gradient, axis=1)
            )}
        )

    def gradient_mask(self, threshold=1):
        values = self.data.mag_gradient
        sets = copy.copy(values.transpose('time', 'lat', 'lon').values)
        for tt in range(len(self.data.time)):
            sets[tt, :, :] = self.data.mag_gradient.isel(time=tt) >= threshold
        sets = xa.DataArray(sets,
                            coords=[self.data.time,
                                    self.data.lat,
                                    self.data.lon],
                            dims=['time', 'lat', 'lon'])
        self.data['gradient_mask'] = sets
        self.data.mag_gradient.values = self.data.mag_gradient.where(
            self.data.gradient_mask
        ).values
        self.data.angle_gradient.values = self.data.angle_gradient.where(
            self.data.gradient_mask
        ).values

    def add_phi(self):
        expectedAngle = np.float32(np.arctan2(
            self.data['lat'] - self.data['LATCENTER'] -
            self.data['centered_lat'] * 0.04 + self.data['LATSHIFT'],
            self.data['lon'] - self.data['LONCENTER'] -
            self.data['centered_lon'] * 0.04 + self.data['LONSHIFT']
        )).transpose((1,0,2))
        self.data = self.data.assign(
            {'phi': (['time', 'lat', 'lon'],
                     wrap_angle_pm_pi(
                         expectedAngle - np.float32(self.data['angle_gradient'])
                     )
            )}
        )
        self.data = self.data.transpose('lat', 'lon', 'time')

    def add_deviation_angle(self):
        self.data = self.data.assign(
            {'deviation_angle': np.abs(
                wrap_angle_pm_pi(self.data['phi'] + np.pi / 2)
            ) - np.pi / 2}
        )

    def compute_radial_profile(self, innerRad=0, outerRad=600, resolution=5,
                               stride=5, variable='temperature',
                               function=np.nanmean, outName='profile',
                               mask=False, nsector=4, start=0):
        shift = (innerRad + outerRad) / 2
        profiles = np.zeros((int((outerRad - innerRad) / resolution),
                             nsector,
                             len(self.data.time.values)))

        angles = np.arctan2(
            self.data['lat'] - self.data['LATCENTER'] -
            self.data['centered_lat'] * 0.04 + self.data['LATSHIFT'],
            self.data['lon'] - self.data['LONCENTER'] -
            self.data['centered_lon'] * 0.04 + self.data['LONSHIFT']
        )
        step = np.pi * 2 / nsector
        centermat = np.zeros((nsector, len(self.data.time.values)))

        for ii in range(len(self.data.time.values)):
            time = self.data.time.values[ii]
            if not isinstance(start, int):
                centers = [start[ii] + step * (jj + 0.5) for jj in range(nsector)]
            else:
                centers = [start + step * (ii + 0.5) for ii in range(nsector)]
            centermat[:, ii] = centers

            radii_mat = self.data.sel(time=time)['radius'].where(
                np.abs(self.data.sel(time=time)['radius'] - shift) <= (outerRad - shift)
            )
            radii = ceil_to_base(radii_mat.values)
            radii = radii[~np.isnan(radii)]
            radii = np.unique(radii)
            if mask:
                radii_mat = radii_mat.where(self.data.sel(time=time)['mask'])
            radii_mat = radii_mat.values

            values = self.data.sel(time=time)[variable].where(
                np.abs(self.data.sel(time=time)['radius'] - shift) <= (outerRad - shift)
            )
            if mask:
                values = values.where(self.data.sel(time=time)['mask'])
            values = values.values

            angle_mat = angles.sel(time=time).where(
                np.abs(self.data.sel(time=time)['radius'] - shift) <= (outerRad - shift)
            ).values
            with np.errstate(invalid='ignore'):
                masks = [(
                        np.abs(np.angle(np.exp(1j*(angle_mat - centers[ii])))) < step/2
                ) for ii in range(nsector)]

            def radfun(x):
                outs = np.zeros(nsector)
                with np.errstate(invalid='ignore'):
                    for ii in range(nsector):
                        outs[ii] = function(
                            values[
                                (x - stride / 2 <= radii_mat) &
                                (radii_mat <= x + stride / 2) &
                                masks[ii]
                            ]
                        )
                return outs

            profiles[:, :, ii] = np.array([radfun(x) for x in radii])

        return xa.Dataset({
            outName: (['radius', 'sector', 'time'], profiles),
            'sectorCenters': (['sector', 'time'], centermat)
        }, coords={'radius': radii,
                   'time': self.data.time.values,
                   'sector': [ii+1 for ii in range(nsector)]})

    def compute_cumulative_profile(self, innerRad=0, outerRad=600, resolution=5,
                                   variable='temperature', function=np.nanmean,
                                   outName='profile', mask=False,
                                   nsector=4, start=0):
        shift = (innerRad + outerRad) / 2
        profiles = np.empty((int((outerRad - innerRad) / resolution),
                             nsector,
                             len(self.data.time.values)))

        angles = np.arctan2(
            self.data['lat'] - self.data['LATCENTER'] -
            self.data['centered_lat'] * 0.04 + self.data['LATSHIFT'],
            self.data['lon'] - self.data['LONCENTER'] -
            self.data['centered_lon'] * 0.04 + self.data['LONSHIFT']
        )
        step = np.pi * 2 / nsector
        centermat = np.zeros((nsector, len(self.data.time.values)))

        for ii in range(len(self.data.time.values)):
            time = self.data.time.values[ii]
            if not isinstance(start, int):
                centers = [start[ii] + step * (jj + 0.5) for jj in range(nsector)]
            else:
                centers = [start + step * (ii + 0.5) for ii in range(nsector)]
            centermat[:, ii] = centers

            radii_mat = self.data.sel(time=time)['radius'].where(
                np.abs(self.data.sel(time=time)['radius'] - shift) <= (outerRad - shift)
            )
            radii = ceil_to_base(radii_mat.values)
            radii = radii[~np.isnan(radii)]
            radii = np.unique(radii)
            if mask:
                radii_mat = radii_mat.where(self.data.sel(time=time)['mask'])
            radii_mat = radii_mat.values

            values = self.data.sel(time=time)[variable].where(
                np.abs(self.data.sel(time=time)['radius'] - shift) <= (outerRad - shift)
            )
            if mask:
                values = values.where(self.data.sel(time=time)['mask'])
            values = values.values

            angle_mat = angles.sel(time=time).where(
                np.abs(self.data.sel(time=time)['radius'] - shift) <= (outerRad - shift)
            ).values
            with np.errstate(invalid='ignore'):
                masks = [(
                        np.abs(np.angle(np.exp(1j * (angle_mat - centers[ii])))) < step / 2
                ) for ii in range(nsector)]

            def radfun(x):
                outs = np.zeros(nsector)
                with np.errstate(invalid='ignore'):
                    for ii in range(nsector):
                        outs[ii] = function(
                            values[(radii_mat <= x) & masks[ii]]
                        )
                return outs

            profiles[:, :, ii] = np.array([radfun(x) for x in radii])

        return xa.Dataset({
            outName: (['radius', 'sector', 'time'], profiles),
            'sectorCenters': (['sector', 'time'], centermat)
        }, coords={'radius': radii,
                   'time': self.data.time.values,
                   'sector': [ii + 1 for ii in range(nsector)]})

    def compute_mean_profile(self, innerRad=0, outerRad=600, resolution=5,
                             stride=5, variable='temperature', angular=False,
                             mask=False, nsector=4, start=0):
        if angular:
            meanfun = angular_mean
        else:
            meanfun = np.nanmean
        return self.compute_radial_profile(innerRad=innerRad,
                                           outerRad=outerRad,
                                           resolution=resolution,
                                           variable=variable,
                                           function=meanfun,
                                           stride=stride,
                                           mask=mask,
                                           outName=variable + '_mean',
                                           nsector=nsector,
                                           start=start
        )

    def compute_variance_profile(self, innerRad=0, outerRad=600, resolution=5,
                                 stride=5, variable='temperature', angular=False,
                                 mask=False, nsector=4, start=0):
        if angular:
            varfun = angular_var
        else:
            varfun = np.nanvar
        return self.compute_radial_profile(innerRad=innerRad,
                                           outerRad=outerRad,
                                           resolution=resolution,
                                           mask=mask,
                                           variable=variable,
                                           function=varfun,
                                           stride=stride,
                                           outName=variable + '_var',
                                           nsector=nsector,
                                           start=start)

    def compute_DAV(self, innerRad=0, outerRad=600, resolution=5,
                    nsector=4, start=0, mask=False):
        return self.compute_cumulative_profile(innerRad=innerRad,
                                               outerRad=outerRad,
                                               resolution=resolution,
                                               variable='deviation_angle',
                                               function=np.nanvar,
                                               mask=mask,
                                               outName='DAV',
                                               nsector=nsector,
                                               start=start)

    def compute_DARV(self, innerRad=0, outerRad=600, resolution=5,
                     nsector=4, start=0, mask=False):
        DARV = self.compute_variance_profile(variable='deviation_angle',
                                             innerRad=innerRad,
                                             outerRad=outerRad,
                                             resolution=resolution,
                                             mask=mask,
                                             nsector=nsector,
                                             start=start)
        mean = self.compute_mean_profile(variable='deviation_angle',
                                         innerRad=innerRad,
                                         outerRad=outerRad,
                                         resolution=resolution,
                                         mask=mask,
                                         nsector=nsector,
                                         start=start)
        N = self.compute_radial_profile(variable='mask',
                                        innerRad=innerRad,
                                        outerRad=outerRad,
                                        resolution=resolution,
                                        function=np.nansum,
                                        nsector=nsector,
                                        start=start)
        DARV = DARV.rename({'deviation_angle_var': 'DARV_ring'})
        DARVx = DARV.DARV_ring.values
        dA = N.profile.values
        DARV = DARV.assign({
            'DARV_N_ring': (('radius', 'sector', 'time'), dA)
        })
        DARVd = np.nancumsum(np.multiply(DARVx, dA), axis=0)
        DARVd = np.divide(DARVd, np.nancumsum(dA, axis=0))
        DARV = DARV.assign({
            'DARV_disk': (('radius', 'sector', 'time'), DARVd)
        })
        DARV = DARV.assign({
            'DARV_mean': (('radius', 'sector', 'time'),
                          mean.deviation_angle_mean.values)
        })

        return DARV

    def compute_DACV(self, innerRad=0, outerRad=600, resolution=5,
                     nsector=4, start=0, mask=False):
        DACV = self.compute_variance_profile(variable='phi',
                                             angular=True,
                                             innerRad=innerRad,
                                             outerRad=outerRad,
                                             resolution=resolution,
                                             mask=mask,
                                             nsector=nsector,
                                             start=start)
        mean = self.compute_mean_profile(variable='phi',
                                         angular=True,
                                         innerRad=innerRad,
                                         outerRad=outerRad,
                                         resolution=resolution,
                                         mask=mask,
                                         nsector=nsector,
                                         start=start)
        N = self.compute_radial_profile(variable='mask',
                                        innerRad=innerRad,
                                        outerRad=outerRad,
                                        resolution=resolution,
                                        function=np.nansum,
                                        nsector=nsector,
                                        start=start)
        DACV = DACV.rename({'phi_var': 'DACV_ring'})
        DACVx = DACV.DACV_ring.values
        dA = N.profile.values
        DACV = DACV.assign({
            'DACV_N_ring': (('radius', 'sector', 'time'), dA)
        })
        DACVd = np.nancumsum(np.multiply(DACVx, dA), axis=0)
        DACVd = np.divide(DACVd, np.nancumsum(dA, axis=0))
        DACV = DACV.assign({
            'DACV_disk': (('radius', 'sector', 'time'), DACVd)
        })
        DACV = DACV.assign({
            'DACV_mean': (('radius', 'sector', 'time'),
                          mean.phi_mean.values)
        })

        return DACV

    def compute_size(self, lowerLimit=-100, upperLimit=0, mask=False,
                     variable='temperature', n=0, rev=False, nanval=100,
                     interp=True, nsector=4, start=0):
        if rev:
            interped = -self.data[variable].transpose('time', 'lat', 'lon').values
        else:
            interped = self.data[variable].transpose('time', 'lat', 'lon').values

        if interp:
            for tt in range(len(self.data.time.values)):
                tmp = copy.copy(interped[tt, :, :])
                tmp[np.isnan(tmp)] = blur(tmp, sigmaX=2, sigmaY=2)[np.isnan(tmp)]
                interped[tt, :, :] = copy.copy(tmp)
        if mask:
            interped = np.where(
                self.data['mask'].transpose('time', 'lat', 'lon'),
                interped,
                nanval
            )

        angles = np.arctan2(
            self.data['lat'] - self.data['LATCENTER'] -
            self.data['centered_lat'] * 0.04 + self.data['LATSHIFT'],
            self.data['lon'] - self.data['LONCENTER'] -
            self.data['centered_lon'] * 0.04 + self.data['LONSHIFT']
        )
        step = np.pi * 2 / nsector
        centermat = np.zeros((nsector, len(self.data.time.values)))

        levels = np.nan_to_num(interped, nan=nanval)
        levels = np.where(levels < lowerLimit, lowerLimit, levels)

        if n == 0: # default to integers
            bins = range(lowerLimit, upperLimit+1)
        else:
            bins = np.linspace(lowerLimit, upperLimit, n+1)

        size = np.zeros((len(self.data.time.values), nsector, len(bins)-1))

        for ii in range(len(self.data.time.values)):
            if not isinstance(start, int):
                centers = [start[ii] + step * (jj + 0.5) for jj in range(nsector)]
            else:
                centers = [start + step * (ii + 0.5) for ii in range(nsector)]
            centermat[:, ii] = centers

            angle_mat = angles.isel(time=ii).values
            with np.errstate(invalid='ignore'):
                masks = [(
                    np.abs(np.angle(np.exp(1j * (angle_mat - centers[ii])))) < step / 2
                ) for ii in range(nsector)]
            for jj in range(nsector):
                size[ii, jj, :] = np.histogram(
                    levels[ii, :, :][masks[jj]], bins
                )[0]
        cumulativeSize = np.cumsum(size, axis=2)

        outName = variable+'_size'
        outName2 = variable+'_delta_size'
        return xa.Dataset({
            outName: (['time', 'sector', 'temp_size'], cumulativeSize),
            outName2: (['time', 'sector', 'temp_size'], size),
            'sectorCenters': (['sector', 'time'], centermat)
        }, coords={
            'temp_size': np.array(
                [ii for ii in bins[:-1]]
            ),
            'time': self.data.time.values,
            'sector': [ii + 1 for ii in range(nsector)]
        })

    def compute_skew(self, lowerTemp=-100, upperTemp=0, mask=True,
                     variable='temperature', resolution=5):
        interped = self.data[variable].transpose('time', 'lat', 'lon').values
        for tt in range(len(self.data.time.values)):
            tmp = copy.copy(interped[tt, :, :])
            tmp[np.isnan(tmp)] = blur(tmp, sigmaX=2, sigmaY=2)[np.isnan(tmp)]
            interped[tt, :, :] = copy.copy(tmp)
        if mask:
            interped = np.where(
                self.data['mask'].transpose('time', 'lat', 'lon'),
                interped,
                100
            )

        levels = np.nan_to_num(np.ceil(interped), nan=100).astype(int)
        levels = np.where(levels < lowerTemp, lowerTemp, levels)

        skewR = np.zeros((len(self.data.time.values),
                         int(np.ceil((upperTemp - lowerTemp+1)/resolution))))
        skewTheta = np.zeros((len(self.data.time.values),
                             int(np.ceil((upperTemp - lowerTemp+1)/resolution))))

        rads = np.float32(self.data.radius.transpose('time', 'lat', 'lon').values)
        angles = np.float16(np.arctan2(
            self.data['lat'] - self.data['LATCENTER'] -
            self.data['centered_lat'] * 0.04 + self.data['LATSHIFT'],
            self.data['lon'] - self.data['LONCENTER'] -
            self.data['centered_lon'] * 0.04 + self.data['LONSHIFT']
        ).transpose('time', 'lat', 'lon').values)
        def skew(t):
            rBar = np.nanmean(np.where(levels <= t, rads, np.NaN), axis=(1, 2))
            xBar = np.nanmean(np.where(
                levels <= t, rads*np.cos(angles), np.NaN
            ), axis=(1, 2))
            yBar = np.nanmean(np.where(
                levels <= t, rads*np.sin(angles), np.NaN
            ), axis=(1, 2))
            return([
                np.divide(
                    np.sqrt(np.add(np.square(xBar), np.square(yBar))),
                    rBar
                ),
                np.arctan2(yBar, xBar)
            ])

        ii = 0
        for t in range(lowerTemp, upperTemp+1, resolution):
            skewR[:, ii], skewTheta[:, ii] = skew(t)
            ii += 1

        outName = variable + '_skew_mag'
        outName2 = variable + '_skew_angle'
        return xa.Dataset({
            outName: (['time', 'temp_skew'], skewR),
            outName2: (['time', 'temp_skew'], skewTheta)
        }, coords={
            'temp_skew': np.array(
                [ii for ii in range(lowerTemp, upperTemp+1, resolution)]
            ),
            'time': self.data.time.values
        })

    def level_set(self, level, variable='temperature', contiguous=False,
                  ringRad=200):
        dat = self.data[variable].transpose('time', 'lat', 'lon')
        interped = np.zeros(dat.shape, 'int16')
        for tt in range(len(self.data.time.values)):
            tmp = copy.copy(dat.isel(time=tt).values)
            tmp[np.isnan(tmp)] = blur(tmp, sigmaX=2, sigmaY=2)[np.isnan(tmp)]
            interped[tt, :, :] = copy.copy(tmp)
        tmp = copy.copy(self.data['radius'].transpose(
            'time', 'lat', 'lon'
        ).where(
            interped <= level
        ).fillna(-999))
        sets = copy.copy(tmp.values)
        if not contiguous:
            sets = -999 < tmp <= level
        else:
            for tt in range(len(tmp.time.values)):
                print('Restricting sets for stamp ' + str(tt + 1) + ' of ' +
                      str(len(tmp.time.values)) + '.')
                g = Graph(sets.shape[1], sets.shape[2], sets[tt, :, :])
                sets[tt, :, :] = g.mark_valid_islands(ringRad)
            sets = xa.DataArray(sets,
                                coords=[self.data.time, self.data.lat, self.data.lon],
                                dims=['time', 'lat', 'lon'])

        return sets

    def add_mask(self, level, variable='temperature', ringRad=200):
        tmp = self.level_set(level, variable=variable, ringRad=ringRad,
                             contiguous=True)
        self.data['mask'] = tmp

    def output_ORB_functions(self, blurRadius=50,
                             DAVinnerRad=0, DAVouterRad=600,
                             DAVresolution=5, RADresolution=5,
                             RADinnerRad=0, RADouterRad=600, stride=5,
                             threshold=1, ringRad=200, nsector=4, start=0,
                             lowerTemp=-100, upperTemp=0,
                             center=True, mask=True):
        if not hasattr(self.data, 'radius'):
            print('Adding radius.')
            self.add_radius()
        if center:
            print('Centering...')
            self.center_on_eye()
        if not hasattr(self.data, 'angle_gradient'):
            print('Adding gradients.')
            self.add_gradient(blurRadius=blurRadius)
        if not hasattr(self.data, 'divergence'):
            print('Adding deviation angles.')
            self.add_divergence()
        if mask and not hasattr(self.data, 'mask'):
            print('Adding mask.')
            self.add_mask(0, ringRad=ringRad)
            self.gradient_mask(threshold=threshold)
        else:
            self.data['mask'] = 1*np.isfinite(self.data.temperature)
        if not hasattr(self.data, 'phi'):
            print('Adding phi.')
            self.add_phi()
        if not hasattr(self.data, 'deviation_angle'):
            print('Adding deviation angles.')
            self.add_deviation_angle()

        print('Computing DAV functions.')
        DAV = self.compute_DAV(innerRad=DAVinnerRad, outerRad=DAVouterRad,
                               resolution=DAVresolution,
                               nsector=nsector, start=start, mask=mask)
        print('Computing DARV functions.')
        DARV = self.compute_DARV(innerRad=DAVinnerRad, outerRad=DAVouterRad,
                                 resolution=DAVresolution,
                                 nsector=nsector, start=start, mask=mask)
        print('Computing DACV functions.')
        DACV = self.compute_DACV(innerRad=DAVinnerRad, outerRad=DAVouterRad,
                                 resolution=DAVresolution,
                                 nsector=nsector, start=start, mask=mask)
        print('Computing RAD functions.')
        RAD = self.compute_radial_profile(innerRad=RADinnerRad,
                                          outerRad=RADouterRad,
                                          resolution=RADresolution,
                                          stride=stride,
                                          nsector=nsector, start=start,
                                          mask=mask)
        RAD = RAD.rename({'profile': 'RAD'})
        print('Computing SIZE functions.')
        SIZE = self.compute_size(lowerLimit=lowerTemp, upperLimit=upperTemp,
                                 nsector=nsector, start=start)
        print('Computing SKEW functions.')
        SKEW = self.compute_skew(lowerTemp=lowerTemp, upperTemp=upperTemp)
        print('Computing IRD functions.')
        IRD = self.compute_radial_profile(variable='divergence', resolution=5,
                                         stride=5, outName='IRD',
                                         nsector=nsector, start=start)

        funcs = DAV.merge(DARV).merge(DACV).merge(RAD).merge(SIZE)
        funcs = funcs.merge(SKEW).merge(IRD)

        return (funcs)

    ### Display related functions
    def plot(self, innerRad=0, outerRad=800, variable='temperature', time=0,
             colRange=(-100, 30), colOption='none'):
        if colOption == 'angle':
            mapOption = mpl.colors.LinearSegmentedColormap.from_list(
                'cyclic_viridis',
                [(0, plt.cm.viridis.colors[0]),
                 (0.25, plt.cm.viridis.colors[256 // 2]),
                 (0.5, plt.cm.viridis.colors[-1]),
                 (1.0, plt.cm.viridis.colors[0])]
            )
        else:
            mapOption = plt.cm.magma
        # Code to display the GOES IR stamp
        self.mask(
            innerRad=innerRad, outerRad=outerRad, variable=variable, time=time
        ).plot(
            cmap=mapOption, vmin=colRange[0], vmax=colRange[1]
        )
        plt.axis('equal')

    def angle_plot(self, variable, innerRad=0, outerRad=800, time=0):
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        with np.errstate(invalid='ignore'):
            ax.scatter(self.data[variable].isel(time=time).where((self.data.isel(time=time)['radius'] <= outerRad) &
                                                                 (self.data.isel(time=time)[
                                                                      'radius'] >= innerRad)).values,
                       self.data['radius'].isel(time=time).where((self.data.isel(time=time)['radius'] <= outerRad) &
                                                                 (self.data.isel(time=time)[
                                                                      'radius'] >= innerRad)).values,
                       marker='.', c='royalblue', s=.02)
        ax.set_ylim(0, outerRad)
        ax.set_rlabel_position(225)
        return ax

    def animate_angles(self, variable, innerRad=0, outerRad=800):
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        s = ax.scatter([], [], marker='.', c='royalblue', s=.02)
        ax.set_ylim(0, outerRad)
        ax.set_rlabel_position(225)

        def updatePoints(tt):
            with np.errstate(invalid='ignore'):
                newdata = np.vstack((
                    self.data[variable].isel(time=tt).where((self.data.isel(time=tt)['radius'] <= outerRad) &
                                                            (self.data.isel(time=tt)[
                                                                 'radius'] >= innerRad)).values.flatten(),
                    self.data['radius'].isel(time=tt).where((self.data.isel(time=tt)['radius'] <= outerRad) &
                                                            (self.data.isel(time=tt)[
                                                                 'radius'] >= innerRad)).values.flatten()
                )).transpose()
                s.set_offsets(newdata)
            return s,

        ani = animation.FuncAnimation(fig, updatePoints, interval=100,
                                      save_count=len(self.data.time.values))
        return ani

    def animate_DAV(self, outerRad=600):
        print('Computing DAV.')
        DAV = self.compute_DAV(innerRad=0, outerRad=outerRad, resolution=5)
        DAV.DAV.values = DAV.DAV.values * (180 ** 2) / (np.pi ** 2)
        print('Computing DAVr.')
        DAVr = self.compute_variance_profile(outerRad=outerRad, variable='deviation_angle')
        DAVr.deviation_angle_var.values = DAVr.deviation_angle_var.values * (180 ** 2) / (np.pi ** 2)
        print('Computing angular means.')
        meanphi = self.compute_mean_profile(innerRad=0, outerRad=600, resolution=5,
                                            angular=True, stride=5, variable='phi')
        print('Computing DACV.')
        varphi = self.compute_variance_profile(innerRad=0, outerRad=600, resolution=5,
                                               angular=True, stride=5, variable='phi')

        # Find worst-case bounding box
        latmax = np.max(self.data.LATSHIFT.values)
        bounds = GOES.coord_from_distance(latmax, 0, outerRad)
        bounds['latHi'] = bounds['latHi'] - latmax
        bounds['latLo'] = bounds['latLo'] - latmax
        idx = {
            'ymin': max([x for x in range(len(self.data.lon)) if self.data.lon[x] <= bounds['lonLo']]),
            'ymax': min([x for x in range(len(self.data.lon)) if self.data.lon[x] >= bounds['lonHi']]),
            'xmin': max([x for x in range(len(self.data.lat)) if self.data.lat[x] <= bounds['latLo']]),
            'xmax': min([x for x in range(len(self.data.lat)) if self.data.lat[x] >= bounds['latHi']])
        }

        fig = plt.figure(figsize=(15, 7))
        # Temperature
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(self.mask(variable='temperature', time=0, outerRad=outerRad).values[
                         idx['xmin']:idx['xmax'], idx['ymin']:idx['ymax']
                         ], extent=[bounds['lonLo'], bounds['lonHi'], bounds['latLo'], bounds['latHi']],
                         cmap=plt.cm.magma, vmin=-100, vmax=30)
        plt.xlabel('Rel. Lon')
        plt.ylabel('Rel. Lat')
        plt.colorbar(im1)

        # Angles
        cyclemap = mpl.colors.LinearSegmentedColormap.from_list(
            'cyclic_viridis',
            [(0, plt.cm.viridis.colors[0]),
             (0.25, plt.cm.viridis.colors[256 // 2]),
             (0.5, plt.cm.viridis.colors[-1]),
             (1.0, plt.cm.viridis.colors[0])]
        )
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(self.mask(variable='deviation_angle', time=0, outerRad=outerRad).values[
                         idx['xmin']:idx['xmax'], idx['ymin']:idx['ymax']
                         ], extent=[bounds['lonLo'], bounds['lonHi'], bounds['latLo'], bounds['latHi']],
                         cmap=cyclemap, vmin=-np.pi / 2, vmax=np.pi / 2)
        plt.xlabel('Rel. Lon')
        plt.ylabel('Rel. Lat')
        plt.colorbar(im2)

        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(self.mask(variable='phi', time=0, outerRad=outerRad).values[
                         idx['xmin']:idx['xmax'], idx['ymin']:idx['ymax']
                         ], extent=[bounds['lonLo'], bounds['lonHi'], bounds['latLo'], bounds['latHi']],
                         cmap=cyclemap, vmin=-np.pi, vmax=np.pi)
        plt.xlabel('Rel. Lon')
        plt.ylabel('Rel. Lat')
        plt.colorbar(im3)

        # Phi scatter
        ax4 = fig.add_subplot(2, 3, 4, polar=True)
        s = ax4.scatter([], [], marker='.', c='royalblue', s=.01)
        ax4.set_ylim(0, outerRad)
        ax4.set_rlabel_position(225)
        l1 = ax4.plot([], [], alpha=.75, c='r')

        # Accumulated values
        ax5 = fig.add_subplot(2, 3, 6)
        finDAV = DAV.DAV.values
        finDAV = finDAV[~np.isnan(finDAV) & ~np.isinf(finDAV)]
        ax5.set_ylim(0, 1.2 * np.max(finDAV))
        ax5.set_xlim(0, outerRad)
        plt.xlabel('Radius (km)')
        plt.ylabel('Linear Variance')
        ax5b = ax5.twinx()
        ax5b.tick_params(axis='y', labelcolor='r')
        ax5b.set_ylabel('Circular Variance', c='r')
        ax5b.set_ylim(0, 1.2)
        l2a = ax5.plot([], [], color='k', label='DAV(r)')
        l2b = ax5.plot([], [], 'k--', label='DARV(r)')
        l2c = ax5b.plot([], [], color='r', label='DACV(r)')
        plt.title('Within-Disk Variance')

        lns = l2a + l2b + l2c
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='upper right')

        # Ring Values
        ax6 = fig.add_subplot(2, 3, 5)
        ax6.set_ylim(0, 1.2 * np.max(finDAV))
        ax6.set_xlim(0, outerRad)
        plt.xlabel('Radius (km)')
        plt.ylabel('Linear Variance')
        ax6b = ax6.twinx()
        ax6b.tick_params(axis='y', labelcolor='r')
        ax6b.set_ylabel('Circular Variance', c='r')
        ax6b.set_ylim(0, 1.2)
        l3a = ax6.plot([], [], color='k', label='DARVr')
        l3b = ax6b.plot([], [], 'r', label='DACVr')
        plt.title('Within-Ring Variance')

        lns = l3a + l3b
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='upper right')

        plt.tight_layout()

        def update_plots(tt):
            # Temperature
            im1.set_array(self.mask(variable='temperature', time=tt, outerRad=outerRad).values[
                          idx['xmin']:idx['xmax'], idx['ymin']:idx['ymax']
                          ])

            # Angles
            im2.set_array(self.mask(variable='deviation_angle', time=tt, outerRad=outerRad).values[
                          idx['xmin']:idx['xmax'], idx['ymin']:idx['ymax']
                          ])

            # Phi
            im3.set_array(self.mask(variable='phi', time=tt, outerRad=outerRad).values[
                          idx['xmin']:idx['xmax'], idx['ymin']:idx['ymax']
                          ])

            # Phi scatter
            with np.errstate(invalid='ignore'):
                newphi = np.vstack((
                    self.data['phi'].isel(time=tt).where(
                        self.data.isel(time=tt)['radius'] <= outerRad).values.flatten(),
                    self.data['radius'].isel(time=tt).where(
                        self.data.isel(time=tt)['radius'] <= outerRad).values.flatten()
                )).transpose()
            s.set_offsets(newphi)
            phi = meanphi.isel(time=tt).phi_mean.values
            r = meanphi.isel(time=tt).radius.values
            x = np.interp(np.linspace(0, outerRad, 2 * outerRad), r, np.cos(phi))
            y = np.interp(np.linspace(0, outerRad, 2 * outerRad), r, np.sin(phi))
            phi = np.arctan2(y, x)
            l1[0].set_data(phi, np.linspace(0, outerRad, 2 * outerRad))

            # DAV updates
            l2a[0].set_data(DAV.isel(time=tt).radius.values,
                            DAV.isel(time=tt).DAV.values)
            r = DAVr.isel(time=tt).radius.values
            x1 = DAVr.isel(time=tt).deviation_angle_var.values
            x2 = varphi.isel(time=tt).phi_var.values
            l2b[0].set_data(r, np.cumsum(x1 * (r / r[0])) / ((r / r[0]) ** 2))
            l2c[0].set_data(r, np.cumsum(x2 * (r / r[0])) / ((r / r[0]) ** 2))

            # DACV updates

            r = varphi.isel(time=tt).radius.values
            l3a[0].set_data(r, x1)
            l3b[0].set_data(r, x2)

        ani = animation.FuncAnimation(fig, update_plots, interval=200,
                                      frames=len(self.data.time.values),
                                      save_count=len(self.data.time.values),
                                      repeat=False)
        return ani

    def mask(self, innerRad=0, outerRad=800, variable='temperature', time=0):
        # Code to provide circular masked IR data
        shift = (innerRad + outerRad) / 2
        return self.data.isel(time=time)[variable].where(
            np.abs(GOES.distance_from_coord(
                self.data.isel(time=time).LATCENTER,
                self.data.coords['lat'] + self.data.isel(time=time).LATSHIFT,
                self.data.isel(time=time).LONCENTER,
                self.data.coords['lon'] + self.data.isel(time=time).LONSHIFT) - shift) <=
            (outerRad - shift)
        )


class ORBbasin:
    def __init__(self, hurdat, threshold=50):
        self.hurdat = hurdat
        self.data = {}
        self.threshold = threshold
        self.PCA = {}
        self.eofs = None

    def add_TC(self, ID, filepath='data/ORB/$ID$-ORB.nc'):
        path = re.sub('\$ID\$', ID, filepath)
        TC = xa.load_dataset(path)

        if hasattr(TC, 'WIND'):
            self.data[ID] = copy.copy(TC)
        else:
            times = TC.sel(
                time=minimum_wind(ID, self.hurdat, self.threshold)
            ).time.values
            wind = np.zeros(times.shape)
            ii = 0
            for time in times:
                wind[ii] = self.hurdat.linear_interp(
                    ID, pd.to_datetime(time)
                ).WIND.values[0]
                ii += 1
            tmp = xa.Dataset({
                'WIND': ('time', wind)
            }, coords={
                'time': times
            })
            self.data[ID] = copy.copy(TC).merge(tmp)

    def access_var_matrix(self, varname='WIND', TC=None,
                          domainMin=None, domainMax=None):
        if TC is None:
            entries = 0
            for id in self.data.keys():
                entries += self.data[id].time.__len__()

            coord = list(self.data[id][varname].dims)
            if coord.__len__() == 1:
                cols = 1
                domain = None
            else:
                dindex = [ii for ii in range(len(coord))
                          if coord[ii] not in ['time', 'sector']][0]
                domain = self.data[id][coord[dindex]].values
                if domainMin is not None:
                    upper_idx = np.where(domain >= domainMin)
                else:
                    upper_idx = np.where(domain >= -1e16)
                if domainMax is not None:
                    lower_idx = np.where(domain <= domainMax)
                else:
                    lower_idx = np.where(domain <= 1e16)
                idx = np.intersect1d(upper_idx, lower_idx)
                cols = idx.__len__()

            if domainMin is not None and domain is None:
                raise SyntaxError('Domain limit defined for variable without '
                                  'domain.')
            elif domainMax is not None and domain is None:
                raise SyntaxError('Domain limit defined for variable without '
                                  'domain.')

            out = np.zeros((entries, cols))
            ii = 0
            for id in self.data.keys():
                n = self.data[id].time.__len__()
                if cols == 1:
                    out[ii:(ii + n), :] = self.data[id][varname].values[:, np.newaxis]
                else:
                    if coord[0] == 'time':
                        tmp = self.data[id][varname].values[:, idx]
                    else:
                        tmp = self.data[id][varname].values[idx, :].transpose((1, 0))
                    out[ii:(ii + n), :] = tmp
                ii += n
        else:
            coord = list(self.data[TC][varname].dims)
            if coord.__len__() == 1:
                out = self.data[TC][varname].values
            else:
                domain = self.data[TC][coord[0]].values
                if domainMin is not None:
                    upper_idx = np.where(domain >= domainMin)
                else:
                    upper_idx = np.where(domain >= -1e16)
                if domainMax is not None:
                    lower_idx = np.where(domain <= domainMax)
                else:
                    lower_idx = np.where(domain <= 1e16)
                idx = np.intersect1d(upper_idx, lower_idx)

                out = self.data[TC][varname].values[idx, :].transpose((1, 0))

        return (out)

    def __incomplete__(self, varname='WIND', TC=None,
                       domainMin=None, domainMax=None):
        return (np.isnan(self.access_var_matrix(
            varname, TC, domainMin, domainMax
        )).any(axis=1))

    def __domain__(self, coord='radius', domainMin=None, domainMax=None,
                   return_idx=False):
        id = list(self.data)[0]
        domain = self.data[id][coord].values
        if domainMin is not None:
            upper_idx = np.where(domain >= domainMin)
        else:
            upper_idx = np.where(domain >= -1e16)
        if domainMax is not None:
            lower_idx = np.where(domain <= domainMax)
        else:
            lower_idx = np.where(domain <= 1e16)
        if return_idx:
            return (np.intersect1d(upper_idx, lower_idx))
        else:
            return (domain[np.intersect1d(upper_idx, lower_idx)])

    def pca(self, varname, coord, d=None, domainMin=None, domainMax=None):
        pca = PCA(n_components=d)
        funcs = self.access_var_matrix(
            varname, domainMin=domainMin, domainMax=domainMax
        )[~self.__incomplete__(
            varname, domainMin=domainMin, domainMax=domainMax
        )]
        center = np.mean(funcs, axis=0)
        centered = funcs - center[np.newaxis, :]

        self.PCA[varname] = pca.fit(centered)

        tmp = xa.Dataset(
            {varname + '_center': ([coord], center)},
            coords={coord: self.__domain__(coord, domainMin, domainMax)}
        )
        if d is None:
            d = self.PCA[varname].components_.shape[0]
        for i in range(d):
            tmp = tmp.assign({
                varname + '_' + str(i + 1): ([coord], self.PCA[varname].components_[i, :])
            })
        if self.eofs is None:
            self.eofs = copy.copy(tmp)
        else:
            self.eofs = self.eofs.merge(copy.copy(tmp))
