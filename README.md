# uncertainty
This folder contains code to estimate multi-attribute cumulative prospect theory (MCTP), decision field theory (DFT), latent DFT adopted in the paper "Preferences for Electric Vehicles under Uncertain Charging Prices: An Eye-tracking Study".

Data:
online.csv: the choice, attribute value and demogarphic of the online street-intercept survey;
lab.csv: the choice, attribute value, eye-movement and demogarphic of the lab experiment.

Estimation:
MCPT.py: likelihood-based estimation of the MCPT;
DFT.py: likelihood-based estimation of the DFT without eye-tracking;
DFT eye.py: likelihood-based estimation of the DFT with eye=tracking;
latent DFT.py: likelihood-based estimation of the latent DFT with eye-tracking;

Results:
MCPT.txt: likelihood-based estimation results of the MCPT;
DFT.txt: likelihood-based estimation results of the DFT without eye-tracking;
DFT eye.txt: likelihood-based estimation results of the DFT with eye=tracking;
latent DFT.xtx: likelihood-based estimation results of the latent DFT with eye-tracking.
