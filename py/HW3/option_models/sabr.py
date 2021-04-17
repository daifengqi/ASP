# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import scipy.integrate as spint
from . import normal
from . import bsm
import pyfeng as pf

'''
MC model class for Beta=1
'''


class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)

    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        impvol = np.ones(len(strike))
        price = self.price(strike, spot, texp, sigma)
        for i, K in enumerate(strike):
            impvol[i] = self.bsm_model._impvol_newton(price[i], K, spot, texp)

        return impvol

    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        # set parameters
        N = 101
        n_path = 10000
        dtk = texp/N
        rand = np.random.multivariate_normal(
            mean=[0, 0], cov=[[1, self.rho], [self.rho, 1]], size=[N, n_path])
        W = rand[:, :, 0]
        Z = rand[:, :, 1]

        # simulate volatility
        sigma_path = np.ones([N, n_path])
        sigma = sigma if sigma else self.sigma
        sigma_path[0, :] = np.ones(n_path) * sigma
        for i in range(1, N):
            sigma_path[i, :] = sigma_path[i-1, :] * \
                np.exp(self.vov*np.sqrt(dtk)*Z[i, :]-0.5*self.vov**2*dtk)

        # simulate price
        spot_path = np.ones([N, n_path])
        spot_path[0, :] = np.ones(n_path) * spot
        for i in range(1, N):
            spot_path[i, :] = np.exp(np.log(spot_path[i-1, :])+sigma_path[i-1, :]*np.sqrt(
                dtk)*W[i, :]-0.5*np.power(sigma_path[i-1, :], 2)*dtk)
        spot_T = spot_path[-1, :]

        # calculate payoff
        price = np.ones(len(strike))
        var = np.ones(len(strike))
        for i, K in enumerate(strike):
            price[i] = np.mean(np.fmax(spot_T - K, 0))
        return price


'''
MC model class for Beta=0
'''


class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None

    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)

    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        impvol = np.ones(len(strike))
        price = self.price(strike, spot, texp, sigma)
        for i, K in enumerate(strike):
            impvol[i] = self.normal_model._impvol_Choi2009(
                price[i], K, spot, texp)

        return impvol

    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        # set parameters
        N = 101
        n_path = 10000
        dtk = texp/N
        rand = np.random.multivariate_normal(
            mean=[0, 0], cov=[[1, self.rho], [self.rho, 1]], size=[N, n_path])
        W = rand[:, :, 0]
        Z = rand[:, :, 1]

        # simulate volatility
        sigma_path = np.ones([N, n_path])
        sigma = sigma if sigma else self.sigma
        sigma_path[0, :] = np.ones(n_path) * sigma
        for i in range(1, N):
            sigma_path[i, :] = sigma_path[i-1, :] * \
                np.exp(self.vov*np.sqrt(dtk)*Z[i, :]-0.5*self.vov**2*dtk)

        # simulate price
        spot_path = np.ones([N, n_path])
        spot_path[0, :] = np.ones(n_path) * spot
        for i in range(1, N):
            spot_path[i, :] = spot_path[i-1, :] + \
                sigma_path[i-1, :]*W[i, :]*np.sqrt(dtk)
        spot_T = spot_path[-1, :]

        # calculate payoff
        price = np.ones(len(strike))
        for i, K in enumerate(strike):
            price[i] = np.mean(np.fmax(spot_T - K, 0))
        return price


'''
Conditional MC model class for Beta=1
'''


class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)

    def bsm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        return 0

    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        n_path = 100000
        N = 100

        # simulate sigma
        m = pf.BsmNdMc(self.vov)
        tobs = np.arange(0, N+1) * texp/N
        m.simulate(tobs=tobs, n_path=n_path)
        sigma_path = np.squeeze(m.path)
        sigma_final = sigma_path[-1, :]
        int_var = spint.simps(sigma_path**2, dx=1, axis=0) * texp/N

        # get S(bsm) and sigma(bsm)
        sigma_0 = self.sigma
        sigma_T = sigma_final * sigma_0
        S_bs = spot*np.exp(self.rho/self.vov*(sigma_T-sigma_0) -
                           0.5*(self.rho*sigma_0)**2*texp*int_var)
        sigma_bs = sigma_0 * np.sqrt((1-self.rho**2)*int_var)

        # bsm formula
        disc_fac = np.exp(-texp * self.intr)
        sigma_std = np.maximum(np.array(sigma_bs) *
                               np.sqrt(texp), np.finfo(float).eps)
        spst = ss  # scipy.stats

        price = np.ones(len(strike))
        for i, K in enumerate(strike):
            d1 = np.log(S_bs / K) / sigma_std
            d2 = d1 - 0.5*sigma_std
            d1 += 0.5*sigma_std
            cp = np.array(cp)
            price_k = S_bs * \
                spst.norm.cdf(cp * d1) - K * spst.norm.cdf(cp * d2)
            price_k *= cp * disc_fac
            price[i] = np.mean(price_k)

        return price


'''
Conditional MC model class for Beta=0
'''


class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None

    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)

    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        return 0

    def price(self, strike, spot, texp, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        n_path = 100000
        N = 100

        # simulate sigma
        m = pf.BsmNdMc(self.vov)
        tobs = np.arange(0, N+1) * texp/N
        m.simulate(tobs=tobs, n_path=n_path)
        sigma_path = np.squeeze(m.path)
        sigma_final = sigma_path[-1, :]
        int_var = spint.simps(sigma_path**2, dx=1, axis=0) * texp/N

        # get S(norm) and sigma(norm)
        sigma_0 = self.sigma
        sigma_T = sigma_final * sigma_0
        S_norm = spot + self.rho/self.vov*(sigma_T-sigma_0)
        sigma_norm = sigma_0 * np.sqrt((1-self.rho**2)*int_var)

        # bachelier formula
        df = np.exp(-texp * self.intr)
        fwd = S_norm
        sigma_std = np.maximum(np.array(sigma_norm) *
                               np.sqrt(texp), np.finfo(float).eps)
        spst = ss

        price = np.ones(len(strike))
        for i, K in enumerate(strike):
            d = (fwd - K) / sigma_std
            price_k = df * (cp*(fwd - K)*spst.norm.cdf(cp *
                            d) + sigma_std * spst.norm.pdf(d))
            price[i] = np.mean(price_k)

        return price
