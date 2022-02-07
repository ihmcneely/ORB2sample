# ORB2sample

Authors: Trey McNeely and Galen Vincent

This repository contains code associated with our paper "Detecting Distributional Differences in Labeled Sequence Data with Application to Tropical Cyclone Satellite Imagery" by Trey McNeely, Galen Vincent, Kimberly M Wood, Rafael Izbicki, and Ann B Lee.

# Contents

This repository contains:
1. Code for working with TC data:
  1. hurdat.py - Code for downloading and working with the HURDAT2 databases for the Atlantic and Pacific hurricane basins. Includes implementation of algorithm for identifying rapid intensification events.
  2. GOES.py - Code for downloading and working with TC infrared imagery from MERGIR. In order to download from the database, you will need to create an account on https://disc.gsfc.nasa.gov/ and follow the instructions at https://disc.gsfc.nasa.gov/data-access.
  3. ORB.py - Code for computing ORB functions, including the radial profiles.
  4. demo.py - Example code for using the above 3 files to scrape imagery of a TC and compute radial profiles for export to csv
2. Code for performing our bootstrap test.
  1. twoSampleTest.py - Code for performing our bootstrap test for sequence data. Includes code for ARMA simulations.
  2. testStudy.py - Bash script for performing the AR1 simulation studies in our paper.
  3. TC-test.ipynb - Python notebook for performing our test on TC radial profiles as in our paper.
