#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import twoSampleTest as tst
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, action='store', default=100)
parser.add_argument("--L", type=int, action='store', default=250)
parser.add_argument("--phi", type=float, action='store', default=0.8)
parser.add_argument("--phiprime", type=float, action='store', default=0.8)
parser.add_argument("--theta", type=float, action='store', default=0.0)
parser.add_argument("--delta", type=float, action='store', default=0.5)
parser.add_argument("--gamma", type=float, action='store', default=0.0)
parser.add_argument("--rho", type=float, action='store', default=0.0)
parser.add_argument("--beta", type=float, action='store', default=0.0)
parser.add_argument("--eta", type=float, action='store', default=0.0)
parser.add_argument("--k", type=int, action='store', default=None)
parsed = parser.parse_args()

N = parsed.N
L = parsed.L
phi = parsed.phi
phiprime = parsed.phiprime
theta = parsed.theta
delta = parsed.delta
gamma = parsed.gamma
rho = parsed.rho
beta = parsed.beta
eta = parsed.eta
k = parsed.k

pvals_glob = np.zeros(N) + 999
pvals_loc = None
grid = pd.DataFrame({
        'x': [8*x/2000-4 for x in range(2001)],
    })
for ii in tqdm(range(N), desc='Replications'):
    Ltest = 250
    L2 = 250

    chainsim = tst.test2sample()
    chainsim.simulate_data(L2, L_test=Ltest, phi=phi, phiprime=phiprime,
                           theta=theta, delta=delta, gamma=gamma, rho=rho,
                           beta=beta, eta=eta)

    sim = tst.test2sample()
    sim.simulate_data(L, L_test=Ltest, phi=phi, phiprime=phiprime,
                      theta=theta, delta=delta, gamma=gamma, rho=rho,
                      beta=beta, eta=eta)

    if k is None:
        relabeler = tst.permuter(sim.train_data.Y)
        relabel = '-perm'
    else:
        relabeler = tst.chain(chainsim.train_data.Y, k=k)
        relabel = '-MC_'+str(k)
    sim.test(relabeler, tst.NW_regressor(r='heuristic'),
             pb=False, groupvar=None)
    grid['rhat'+str(ii)] = sim.r0.predict(grid)
    sim.test_data['Replication'] = ii + 1

    if pvals_loc is None:
        pvals_loc = np.zeros((sim.test_data.shape[0]*N,
                              sim.test_data.shape[1]))

    pvals_loc[Ltest*ii:Ltest*(ii+1), :] = sim.test_data.values
    pvals_glob[ii] = sim.get_global()

pvals_loc = pd.DataFrame(data=pvals_loc, columns=sim.test_data.columns)

grid.to_csv('NW-reps_'+str(int(N))+
             '-B_200-L_'+str(int(L))+
             relabel+
             '-phi_'+str(phi)+
             '-phiprime_'+str(phiprime)+
             '-theta_'+str(theta)+
             '-delta_'+str(delta)+
             '-gamma_'+str(gamma)+
             '-rho_'+str(rho)+
             '-eta_'+str(eta)+
             '-beta_'+str(beta)+
             '-reg.csv')
pvals_loc.to_csv('NW-reps_'+str(int(N))+
                 '-B_200-L_'+str(int(L))+
                 relabel+
                 '-phi_'+str(phi)+
                 '-phiprime_'+str(phiprime)+
                 '-theta_'+str(theta)+
                 '-delta_'+str(delta)+
                 '-gamma_'+str(gamma)+
                 '-rho_'+str(rho)+
                 '-eta_'+str(eta)+
                 '-beta_'+str(beta)+
                 '-local.csv')
np.savetxt('NW-reps_'+str(int(N))+
           '-B_200-L_'+str(int(L))+
           relabel+
           '-phi_'+str(phi)+
           '-phiprime_'+str(phiprime)+
           '-theta_'+str(theta)+
           '-delta_'+str(delta)+
           '-gamma_'+str(gamma)+
           '-rho_'+str(rho)+
           '-eta_'+str(eta)+
           '-beta_'+str(beta)+
           '-global.csv', pvals_glob, delimiter=',')