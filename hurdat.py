import numpy as np
import pandas as pd
import io
import urllib.request as url
from datetime import datetime, timedelta

class Hurdat:
    """Class for interfacing with the hurdat2 database."""

    def __init__(self, data=None,
                 nalPath='https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2020-052921.txt',
                 pacPath='https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2020-043021a.txt'):
        """
        Import and parse the Hurdat2 dataset.

        :param nalPath: Filepath or URL to access the Hurdat2 text file for the North Atlantic Basin
        :param pacPath: Filepath or URL to access the Hurdat2 text file for the Eastern North Pacific Basin
        """
        if 'http' in nalPath:
            nalFile = url.urlopen(nalPath)
            nalRaw = nalFile.read().decode().splitlines(keepends=True)
        else:
            nalFile = open(file=nalPath)
            nalRaw = nalFile.read().splitlines(keepends=True)
        if 'http' in pacPath:
            pacFile = url.urlopen(pacPath)
            pacRaw = pacFile.read().decode().splitlines(keepends=True)
        else:
            pacFile = open(file=pacPath)
            pacRaw = pacFile.read().splitlines(keepends=True)
        hurdatRaw = nalRaw + pacRaw
        nalFile.close()
        pacFile.close()

        # Read the headers, get storm IDs, names, and lines comprised.
        ii = 0
        stormHeads = dict()
        colNames = ['DATE', 'TIME', 'STATUS', 'CATEGORY', 'LAT', 'LON', 'WIND',
                    'PRESS', 'trash9', 'trash10', 'trash11', 'trash12',
                    'trash13', 'trash14', 'trash15', 'trash16', 'trash17',
                    'trash18', 'trash19', 'trash20']

        # Collect headers
        for line in hurdatRaw:
            if line[0] in 'AEC':
                id = line[0:8]
                name = line[18:28].strip()
                entries = int(line[34:36].strip())
                startLine = ii + 1
                endLine = ii + entries
                stormHeads[id] = {'id': id,
                                  'name': name,
                                  'entries': entries,
                                  'startLine': startLine,
                                  'endLine': endLine}
            ii += 1

        # collect list of storm names, ids
        ii = 0
        drops = list(i for i in range(len(hurdatRaw)) if hurdatRaw[i][0] in 'AEC')
        drops.append(len(hurdatRaw) - 1)
        entries = np.diff(drops) - 1
        entries[-1] += 1
        ids = list()
        names = list()
        for stormHead in stormHeads.values():
            ids.extend([stormHead['id']] * entries[ii])
            names.extend([stormHead['name']] * entries[ii])
            ii += 1

        # Allocate whole file at once
        storms = (
            pd.read_csv(io.StringIO(''.join(
                list(hurdatRaw[i] for i in range(len(hurdatRaw))
                     if hurdatRaw[i][0] not in 'AEC')
            )),
                names=colNames, dtype=str,
                delimiter=',\s+'
            ).filter(items=colNames[0:7], axis=1
                     ).assign(ID=ids, NAME=names)
        )

        # Combine the data, clean columns
        def lat_lon_fun(strSeries):
            out = [0] * len(strSeries)
            ii = 0
            for string in strSeries:
                string = str(string)
                string = string.strip()
                if string[-1] in 'WS':
                    out[ii] = -float(string[:-1])
                else:
                    out[ii] = float(string[:-1])
                ii += 1
            return out

        storms = (
            storms
                .assign(LAT=lat_lon_fun(storms.LAT),
                        LON=lat_lon_fun(storms.LON),
                        DATETIME=pd.to_datetime(storms.DATE + storms.TIME,
                                                format='%Y%m%d%H%M'))
                .drop(labels='STATUS', axis=1)
        )
        storms = storms[storms['TIME'].astype('int') % 600 == 0]

        storms['WIND'] = pd.to_numeric(storms['WIND'])

        self.storms = storms

    def get_tc_by_id(self, id):
        """
        Extract the Hurdat2 entry for a single TC by its unique ID.

        :param id: Hurdat ID for TC, e.g. AL152016 for Hurricane Nicole [2016] (15th TC in the North Atlantic in the
                   2016 season.
        :return: Pandas dataframe containing full Hurdat2 data for the TC of interest.
        """
        return self.storms[self.storms['ID'].str.match(id)]

    def linear_interp(self, id, time):
        """
        Get estimated Hurdat2 data for TC at non-synoptic times.

        The Hurdat2 database records best track data for TCs at 6 hour intervals. This function provides a linear
        interpolation for TC location and intensity.

        :param id: Hurdat ID for TC, e.g. AL152016 for Hurricane Nicole [2016] (15th TC in the North Atlantic in the
                   2016 season.
        :param time: Time at which TC data is desired.
        :return: Data frame row containing interpolated TC data.
        """
        storm = self.get_tc_by_id(id)
        last_ts = storm[storm['DATETIME'] <= time].iloc[[-1]]
        if last_ts.iloc[0]['DATETIME'] == time:
            return(last_ts)
        next_ts = storm[storm['DATETIME'] >= time].iloc[[0]]
        if next_ts.iloc[0]['DATETIME'] == time:
            return(next_ts)
        fraction = (
                np.abs(time - last_ts.iloc[0]['DATETIME'])/
                (
                    np.abs(next_ts.iloc[0]['DATETIME'] - time) +
                    np.abs(time - last_ts.iloc[0]['DATETIME'])
                )
        )
        out = last_ts
        for col in ['LAT', 'LON', 'WIND']:
            dif = fraction*(next_ts.iloc[0][col] - last_ts.iloc[0][col])
            out[col] = last_ts.iloc[0][col] + dif

        out['DATETIME'] = time
        out['DATE'] = time.strftime(format='%Y%m%d')
        out['TIME'] = time.strftime(format='%H%M')

        return(out)

    def get_hourly_position(self, id):
        """
        Interpolates Hurdat2 data for a TC from 6-hourly to hourly data.

        The Hurdat2 database records best track data for TCs at 6 hour intervals. Uses Hurdat.linear_interp() to
        estimate hourly track data.

        :param id: Hurdat ID for TC, e.g. AL152016 for Hurricane Nicole [2016] (15th TC in the North Atlantic in the
                   2016 season.
        :return: Data frame containing hourly interpolation of TC data.
        """
        storm = self.get_tc_by_id(id)
        startDate = np.min(storm['DATETIME'])
        endDate = np.max(storm['DATETIME'])
        hourly_position = self.linear_interp(id, startDate)
        date = startDate + timedelta(hours=1)
        while date <= endDate:
            hourly_position = hourly_position.append(
                self.linear_interp(id, date),
                ignore_index=True
            )
            date = date + timedelta(hours=1)

        return(hourly_position)

    def get_half_hourly_position(self, id):
        """
        Interpolates Hurdat2 data for a TC from 6-hourly to half hourly data.

        The Hurdat2 database records best track data for TCs at 6 hour intervals. Uses Hurdat.linear_interp() to
        estimate half hourly track data.

        :param id: Hurdat ID for TC, e.g. AL152016 for Hurricane Nicole [2016] (15th TC in the North Atlantic in the
                   2016 season.
        :return: Data frame containing hourly interpolation of TC data.
        """
        storm = self.get_tc_by_id(id)
        startDate = np.min(storm['DATETIME'])
        endDate = np.max(storm['DATETIME'])
        hourly_position = self.linear_interp(id, startDate)
        date = startDate + timedelta(hours=.5)
        while date <= endDate:
            hourly_position = hourly_position.append(
                self.linear_interp(id, date),
                ignore_index=True
            )
            date = date + timedelta(hours=0.5)

        return(hourly_position)

    def genesis_to_lysis_filter(self, minimum_wind):
        """
        Filter's storms based on needing to have wind speed > minimum_wind for
        at least one time step. Then returns all data for the storm between
        the first time it crossed the threshold (genesis) to the last time it
        falls below the threshold (lysis).

        :param minimum_wind: Wind threshold for defining genesis and lysis. Units of kt. 
        :return: A new Hurdat instance containing the subsetted data.
        """

        # Get modifiable copy of dataframe of storms
        hurdat = self.storms.copy()

        # Create a column for the maximum wind seen during each storm
        hurdat['MAX_WIND'] = hurdat.groupby('ID')['WIND'].transform('max')

        # Filter to only have storms that at some point go above min. wind
        hurdat = hurdat.loc[hurdat['MAX_WIND'] >= minimum_wind]

        # Create column containing time of the TC only if it is above the min
        # wind boundary
        hurdat['DATETIME_ABOVE_MIN_WIND'] = hurdat['DATETIME'].where(hurdat['WIND'] >= minimum_wind)
        
        # Calculate the start and stop conditions for each storm based on the 
        # maximum and minimum times where the storm goes above the minimum wind
        hurdat['Start'] = hurdat.groupby('ID')['DATETIME_ABOVE_MIN_WIND'].transform('min')
        hurdat['Stop'] = hurdat.groupby('ID')['DATETIME_ABOVE_MIN_WIND'].transform('max')

        # Filter down dataset to only include storms between genesis and lysis
        hurdat = hurdat.loc[(hurdat['DATETIME'] >= hurdat['Start']) & (hurdat['DATETIME'] <= hurdat['Stop'])]
        hurdat.drop(['MAX_WIND', 'DATETIME_ABOVE_MIN_WIND', 'Start', 'Stop'], axis = 1, inplace = True)
        hurdat = hurdat.reset_index(drop = True)
        

        return Hurdat(data = hurdat)

    def identify_events(self, threshold):
        """
        Identifies each point in the dataset as either being in a rapid
        intensification event, or being in a rapid weakening event. Both are 
        defined as a change in storm wind speed of at least :threshold:kt within
        a 24 hour period.

        :param threshold: Wind speed change (kt) needed to define RI or RW event. 
        :return: Nothing. Just adds two columns to the storms dataframe, one for
                 indication of an RI event, one for indication of a RW event.
        """

        def single_storm_identify(intensities, thresh):
            TT = len(intensities)
            # If storm is to short for the algorithm, don't identify anything
            if TT < 5:
                return np.full(TT, False)

            ZZ = np.full((TT, TT), False)

            del_1 = np.diff(intensities, n = 1) # lag-1 differences
            del_4_mat = np.zeros((5, TT-4)) 
            for ii in range(1, 5):
                del_4_mat[ii,:] = [intensities[jj+ii] - intensities[jj] for jj in range(TT-4)]
            del_4 = np.amax(del_4_mat, axis = 0) # max of lag 0, 1, 2, 3, 4 differences

            AA = np.where(del_4 >= thresh)[0] # All times with RI
            BB = np.where(del_1 > 0)[0] # All increasing lag-1 times

            # Follow the algorithm outlined elsewhere for calculating RI
            for tt in range(TT - 4):
                if np.isin(tt, AA):
                    ZZ[tt, tt:(tt + 5)] = True
                    for hh in range(4, 0, -1):
                        if not np.isin(tt+hh-1, BB):
                            ZZ[tt, tt+hh] = False
                        else:
                            break
                    for hh in range(4):
                        if not np.isin(tt+hh, BB):
                            ZZ[tt, tt+hh] = False
                        else:
                            break
            YY = np.any(ZZ, 0)
            return YY

        # Get list of storm IDs
        ids = pd.unique(self.storms['ID'])

        # Loop procedure over all storms for RI and RW, keeping track of things
        # in order
        RI = []
        RW = []
        for id in ids:
            intensity = np.array(self.get_tc_by_id(id)['WIND'])
            RI_temp = single_storm_identify(intensity, threshold)
            RI.extend(RI_temp)

            RW_temp = np.flip(single_storm_identify(np.flip(intensity), threshold))
            RW.extend(RW_temp)
        
        # Add new columns to dataframe
        self.storms['RI'] = RI
        self.storms['RW'] = RW
    
    def distance_to_land_label(self, min_distance):
        """
        Creates a column in the storms dataframe as a T/F of whether or not the
        storm's center is wihtin min_distance of land. Used to identify which
        events we want to classify as RI/RW events for study. Column is 1 if 
        within min_distance of land, 0 otherwise. 
        """

        assert 'DISTANCE' in self.storms.columns, "Need a DISTANCE column in order to compute this function."

        within_distance = self.storms['DISTANCE'] < min_distance
        self.storms['NEAR_LAND'] = within_distance


        




