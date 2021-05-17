import os, re, pdb, json, argparse

import pandas as pd
# import matplotlib.pyplot as plt
from natsort import natsorted 
import numpy as np

import time
import sys


sys.path.append('../')

def print_result(hvac, coil, ppd):
    print(' HVAC POWER:', hvac, 'BASELINE DIFF:', baseline_hvac - hvac )
    print(' COIL POWER:', coil, 'BASELINE DIFF:', baseline_coil - coil)
    print('        PPD:', ppd,  'BASELINE DIFF:', ppd - baseline_ppd)
    
def load_baseline(pth):
    baseline = pd.read_csv(f'../baselines/{pth}.csv', index_col='time')

    if len(baseline) == 0:
        print('raising value error')
        raise ValueError()
        
    baseline.index = pd.to_datetime(baseline.index)

    if 'heating' in pth:
        baseline = baseline['1991-01-01 00:15:00':'1991-02-01 23:45:00']
    else:
        baseline = baseline['1991-07-01 00:15:00':'1991-08-02 00:00:00']
    
    if len(baseline) == 0:
        print('raising value error 2')
        raise ValueError()
    
    baseline_coil = baseline['Heat Coil Power'].sum()
    baseline_hvac = baseline['HVAC Power'].sum()
    
    comfort = baseline[baseline["Occupancy Flag"]==1].copy()
    
    if len(comfort) == 0:
        print('raising value error 3')
        raise ValueError()
    
    baseline_ppd = comfort["PPD"].mean()
    illum_mean = ((comfort['Illuminance 1'] + comfort['Illuminance 2']) / 2).mean()
    ill = (comfort['Illuminance 1'] + comfort['Illuminance 2']) / 2
    try:
        illum_viol = len(ill[(ill < 300) | (ill > 750)]) / len(comfort)
    except ZeroDivisionError:
        pdb.set_trace()
        
    return baseline_coil, baseline_hvac, baseline_ppd, illum_viol

def load_result(pth):
    root = os.path.join(ROOT, 'rl_results', pth)
    
    rewards = []
    hvacs = []
    hcoils = []
    ccoils = []
    ppds  = []
    illum = []

    x = os.listdir(root)
    for csv in natsorted(x):
        if 'csv' in csv:
            print(csv)
            csv_pth = os.path.join(root, csv)
            df = pd.read_csv(csv_pth, index_col='time', usecols=[
                'time', 'Heat Coil Power', 'Cool Coil Power', 'HVAC Power',
                "Occupancy Flag", 'Illuminance 1', 'Illuminance 2', 'reward',
                'PPD'
            ])
            df.index = pd.to_datetime(df.index)   
            
            if len(df) == 0:
                print('raising value error 2')
                raise ValueError()

            rewards.append(df['reward'].sum())

            df_hcoil = df['Heat Coil Power'].sum()
            df_ccoil = df['Cool Coil Power'].sum()
            df_hvac = df['HVAC Power'].sum()
            
            comfort = df[df["Occupancy Flag"]==1].copy()
            
            if len(comfort) == 0:
                print('raising value error 3:',csv_pth)
                raise ValueError()

            df_ppd = comfort["PPD"].mean()
            illum_mean = ((comfort['Illuminance 1'] + comfort['Illuminance 2']) / 2).mean()
            ill = (comfort['Illuminance 1'] + comfort['Illuminance 2']) / 2
            try:
                illum_viol = len(ill[(ill < 300) | (ill > 750)]) / len(comfort)
            except ZeroDivisionError:
                pdb.set_trace()

            hvacs.append(df_hvac)
            hcoils.append(df_hcoil)
            ccoils.append(df_ccoil)
            ppds.append(df_ppd)
            illum.append(illum_viol)
    return hvacs, hcoils, ccoils, ppds, illum, rewards

hvac_results = {}
hcoils_results = {}
ccoils_results = {}
ppds_results = {}
illum_results = {}
rewards_results = {}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='test')
    
    parser.add_argument('--a', type=str, required=True, help='')
    parser.add_argument('--b', type=str, required=True)
    parser.add_argument('--d', type=str, required=True)
    parser.add_argument('--s', type=str, required=True)
    parser.add_argument('--rp1', type=float, required=True)
    parser.add_argument('--rp2', type=float, required=True)
    parser.add_argument('--rp3', type=float, required=True)
    
    args = parser.parse_args()
    
    a = args.a
    b = args.b
    d = args.d
    s = args.s
    rp1 = args.rp1
    rp2 = args.rp2
    rp3 = args.rp3
    print('======== agent', a)
    print('======== blinds', b) 
    print('======== dlight', d) 
    print('======== season', s) 
    print('======== rp1', rp1) 
    print('======== rp2', rp2) 
    print('======== rp3', rp3)

    if a == 'SAC':
        ROOT = '/scratch/gbaasch/hvac_ctonrol'
        n = 'leaky'
    else:
        ROOT = '/scratch/gbaasch/hvac_control'
        n = 'octo'

    try:
        fname = f'{a}_{n}_{s}_blinds{b}_dlighting{d}_{rp1}_{rp2}_{rp3}'
        hvacs, hcoils, ccoils, ppds, illums, rewards = load_result(fname)
        results = {}
        results['hvac'] = hvacs
        results['hcoils'] = hcoils
        results['ccoils'] = ccoils
        results['ppds'] = ppds
        results['illums'] = illums
        results['rewards'] = rewards

        with open(f"results/{fname}.csv", "wb") as f:
            f.write(json.dumps(results).encode("utf-8"))
#        hvac_results[fname] = hvacs
 #       hcoils_results[fname] = hcoils
  #      ccoils_results[fname] = ccoils
   #     ppds_results[fname] = ppds
    #    illum_results[fname] = illums
    #    rewards_results[fname] = rewards

   #     if not os.path.exists(f"results/{fname}"):
   #         os.makedirs(f"results/{fname}")

   #     with open(f"results/{fname}/hvacs.json", "wb") as f:
   #         f.write(json.dumps(hvac_results).encode("utf-8"))

   #     with open(f"results/{fname}/hcoils.json", "wb") as f:
   #         f.write(json.dumps(hcoils_results).encode("utf-8"))

   #     with open(f"results/{fname}/ccoils.json", "wb") as f:
   #         f.write(json.dumps(ccoils_results).encode("utf-8"))

   #     with open(f"results/{fname}/ppds.json", "wb") as f:
   #         f.write(json.dumps(ppds_results).encode("utf-8"))

   #     with open(f"results/{fname}/illum.json", "wb") as f:
   #         f.write(json.dumps(illum_results).encode("utf-8"))

   #     with open(f"results/{fname}/rewards.json", "wb") as f:
   #         f.write(json.dumps(rewards_results).encode("utf-8"))

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
