import celfa_data as udata
import numpy as np


muon_data = udata.construct_data_array(["charge", "time_abs", "EnergyMuon", "VisibleEnergy"],
                                       "/home/daniel/CNN/beamlike/Muon/",
                                       "muon_beamlike")
electron_data = udata.construct_data_array(["charge", "time_abs", "EnergyElectron", "VisibleEnergy"],
                                           "/home/daniel/CNN/beamlike/Electron/",
                                           "electron_beamlike")

category_identifiers = np.array(udata.build_category_values([(1, 0), (0, 1)],
                                                            len(electron_data) if len(electron_data) < len(muon_data)
                                                            else len(muon_data)))
