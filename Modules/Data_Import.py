from __future__ import division
import pandas
from six.moves import cPickle as pickle

import pandas
import numpy as np
from Modules.HEP_Proc import *

classTrainFeatures = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_met_pt', 'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_all_pt', 'PRI_tau_px', 'PRI_tau_py', 'PRI_tau_pz', 'PRI_lep_px', 'PRI_lep_py', 'PRI_lep_pz', 'PRI_jet_leading_px', 'PRI_jet_leading_py', 'PRI_jet_leading_pz', 'PRI_jet_subleading_px', 'PRI_jet_subleading_py', 'PRI_jet_subleading_pz', 'PRI_met_px', 'PRI_met_py']

def importData(dirLoc = "../Data/",
               rotate=False, cartesian=True, mode='OpenData'):
    '''Import and preprocess data from CSV'''
    if mode == 'OpenData':
        data = pandas.read_csv(dirLoc + 'atlas-higgs-challenge-2014-v2.csv')
        data.rename(index=str, columns={"KaggleWeight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        data.drop(columns=['Weight'], inplace=True) 
        data = data[data.KaggleSet == 't']

    else:
        data = pandas.read_csv(dirLoc + 'training.csv')
        data.rename(index=str, columns={"Weight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)

    convertData(data, rotate, cartesian)

    data['gen_target'] = 0
    data.loc[data.Label == 's', 'gen_target'] = 1
    data.drop(columns=['Label', 'KaggleSet'], inplace=True)   
    trainFeatures = [x for x in data.columns if 'gen' not in x and x != 'EventId']
    print('Training on {} datapoints with {} features:\n{}'.format(len(data), len(trainFeatures), [x for x in trainFeatures]))
    data['gen_norm_weight'] = 0

    return data[trainFeatures + ['gen_target', 'gen_weight', 'gen_norm_weight']], trainFeatures

    # return {'inputs': data[trainFeatures].values.astype('float32'),
    #         'targets': data['gen_target'].values.astype('int'),
    #         'weights': data['gen_weight'].values.astype('float32')}

def rotateEvent(inData):
    '''Rotate event in phi such that lepton is at phi == 0'''
    inData['PRI_tau_phi'] = deltaphi(inData['PRI_lep_phi'], inData['PRI_tau_phi'])
    inData['PRI_jet_leading_phi'] = deltaphi(inData['PRI_lep_phi'], inData['PRI_jet_leading_phi'])
    inData['PRI_jet_subleading_phi'] = deltaphi(inData['PRI_lep_phi'], inData['PRI_jet_subleading_phi'])
    inData['PRI_met_phi'] = deltaphi(inData['PRI_lep_phi'], inData['PRI_met_phi'])
    inData['PRI_lep_phi'] = 0
    
def convertData(inData, rotate=False, cartesian=True):
    '''Pass data through conversions and drop uneeded columns'''
    inData.replace([np.inf, -np.inf], np.nan, inplace=True)
    inData.fillna(-999.0, inplace=True)
    inData.replace(-999.0, 0.0, inplace=True)
    
    if rotate:
        rotateEvent(inData)
    
    if cartesian:
        moveToCartesian(inData, 'PRI_tau', drop=True)
        moveToCartesian(inData, 'PRI_lep', drop=True)
        moveToCartesian(inData, 'PRI_jet_leading', drop=True)
        moveToCartesian(inData, 'PRI_jet_subleading', drop=True)
        moveToCartesian(inData, 'PRI_met', z=False)
        
        inData.drop(columns=["PRI_met_phi"], inplace=True)
        
    if rotate and not cartesian:
        inData.drop(columns=["PRI_lep_phi"], inplace=True)
    elif rotate and cartesian:
        inData.drop(columns=["PRI_lep_py"], inplace=True)

