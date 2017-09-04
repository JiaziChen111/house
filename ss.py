# -*- coding: utf-8 -*-

"""
Oct. 2016, Hyun Chang Yi
Heterogeneous Agent OLG Model
"""

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import norm
from math import sqrt, exp
from numpy import (linspace, mean, array, zeros, absolute, dot, prod, int,
                   sum, argmax, tile, concatenate, ones, log, unravel_index,
                   cumsum, newaxis, maximum, minimum, repeat, cumprod, int64)
from numpy.random import random
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle
import os
from platform import system
from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import Value, Array, RawArray
from ctypes import Structure, c_double


class params:
    # This class is just a "struct" to hold the collection of PARAMETER values
    def __init__(self, T=240, spm=0.1, ALP=0.36, DEL=0.06, PSI=0.1, BET=0.994,
                 SIG=1.5, Tr=0.2378, Tw=0.2378, Th=0.01, TC=0.02, dti=0.5,
                 ltv=0.7, THE=0.1, ZET=0.3, yN=79, W=45, R=34, zN=18,
                 SIGMAe=sqrt(0.045), GAM=0.96, aL=-10.0, aH=60.0, aN=200,
                 hL=0.1, hH=1.0, hN=7, hs=7, SSr=0.5, SSq=0.1, SSs=0.1,
                 gs=1.5, tol=1e-2, tpp=1.5, savgol_windows=71, savgol_order=1):
        self.savgol_windows, self.savgol_order = savgol_windows, savgol_order
        self.SSr, self.SSq, self.SSs, self.tpp = SSr, SSq, SSs, tpp
        self.gs, self.tol = gs, tol

        self.ALP, self.ZET, self.DEL = ALP, ZET, DEL
        self.Tr, self.Tw, self.Th = Tr, Tw, Th
        self.THE, self.PSI = THE, PSI
        self.TC, self.dti, self.ltv = TC, dti, ltv
        self.BET, self.SIG = BET, SIG

        self.zN, self.yN, self.hN = zN, yN, hN
        self.T, self.R, self.W = T, R, W

        """Grid for Liquid Asset"""
        self.aH, self.aL = aH, aL
        ap = aH * linspace(0, 1, aN)**gs
        am = -ap[1:][::-1]
        aa = concatenate((am, ap))
        self.aa = aa[aa > aL]
        self.aN = aN = len(self.aa)
        self.a0_id = aN - len(ap)

        """Grid for House"""
        self.hh = linspace(hL, hH, hN)
        self.hs = hs

        """survival probabilities for initial and terminal steady states"""
        sp0 = array([1.00000, 0.99962, 0.99960, 0.99958, 0.99956, 0.99954,
                     0.99952, 0.99950, 0.99947, 0.99945, 0.99942, 0.99940,
                     0.99938, 0.99934, 0.99930, 0.99925, 0.99919, 0.99910,
                     0.99899, 0.99887, 0.99875, 0.99862, 0.99848, 0.99833,
                     0.99816, 0.99797, 0.99775, 0.99753, 0.99731, 0.99708,
                     0.99685, 0.99659, 0.99630, 0.99599, 0.99566, 0.99529,
                     0.99492, 0.99454, 0.99418, 0.99381, 0.99340, 0.99291,
                     0.99229, 0.99150, 0.99057, 0.98952, 0.98841, 0.98719,
                     0.98582, 0.98422, 0.98241, 0.98051, 0.97852, 0.97639,
                     0.97392, 0.97086, 0.96714, 0.96279, 0.95795, 0.95241,
                     0.94646, 0.94005, 0.93274, 0.92434, 0.91518, 0.90571,
                     0.89558, 0.88484, 0.87352, 0.86166, 0.84930, 0.83652,
                     0.82338, 0.80997, 0.79638, 0.78271, 0.76907, 0.75559,
                     0.74239])
        sp1 = sp0 + spm * (1 - sp0)
        self.sp = concatenate((sp0[newaxis, :], repeat(sp1[newaxis, :], T - 1,
                                                       axis=0)), axis=0)
        self.spr = (1 - self.sp) / self.sp
        self.npg = array([1.005, 1.0048, 1.0046, 1.0075, 1.0045, 1.0043,
                          1.0041, 1.0038, 1.0036, 1.0034, 1.0032, 1.0030,
                          1.0028, 1.0025, 1.0023, 1.0020, 1.0019, 1.0016,
                          1.0013, 1.0010, 1.0007, 1.0004, 1.0001, 0.9997,
                          0.9994, 0.9989, 0.9986, 0.9981, 0.9977, 0.9973,
                          0.9969, 0.9965, 0.9961, 0.9957, 0.9953, 0.9949,
                          0.9945, 0.9942, 0.9938, 0.9935, 0.9931, 0.9928,
                          0.9924, 0.9921, 0.9919, 0.9915, 0.9912, 0.9911,
                          0.9909, 0.9906, 0.9904, 0.9902, 0.9900, 0.9900,
                          0.9902, 0.9904, 0.9906, 0.9909, 0.9911, 0.9912,
                          0.9915, 0.9919, 0.9921, 0.9924, 0.9928, 0.9931,
                          0.9935, 0.9938, 0.9942, 0.9945, 0.9949, 0.9953,
                          0.9957, 0.9961, 0.9965, 0.9969, 0.9973, 0.9977,
                          0.9981, 0.9986, 0.9989, 0.9994, 0.9997, 1.0000,
                          1.0000])
        # self.npg = array([1.005, 1.0000])
        self.npg = concatenate((self.npg, repeat(self.npg[-1],
                                                 T - len(self.npg))))
        self.nbp = cumprod(self.npg)
        # self.nbp = concatenate((nbp,repeat(nbp[-1],T-len(nbp))))

        # new setup of population given the population projection upto 2100
        m0 = array([prod(self.sp[0][:y + 1]) / self.npg[0]**y
                    for y in range(yN)], dtype=float)
        self.pop = array([m0 for t in range(T)], dtype=float)
        for t in range(1, T):
            self.pop[t, 0] = self.nbp[t]
            for y in range(1, yN):
                self.pop[t, y] = self.pop[t - 1, y - 1] * self.sp[t][y]

        # Construct containers for market ps, tax rates, pension, bequest
        self.HS = self.hs * sum(self.pop[0]) * ones(T)
        # self.HS = array([par.hs*sum(self.pop[t]) for t in range(T)])

        """ LOAD PARAMETERS : SURVIVAL PROB., INITIAL DIST. OF PRODUCTIVITY,
        PRODUCTIVITY TRANSITION PROB. AND PRODUCTIVITY """
        self.SIGMAe = SIGMAe
        self.GAM = GAM
        SIGMAy = sqrt(SIGMAe**2 / (1 - GAM**2))
        age = array([0, 15, 35, 50])
        eff = array([0.11931, 0.41017, 0.47964, 0.23370])
        eff = InterpolatedUnivariateSpline(age, eff, k=1)(range(yN))
        age = array([2.5, 10, 20, 30, 40, 50])
        ls = array([0.796, 0.919, 0.919, 0.875, 0.687, 0.19])
        ls = InterpolatedUnivariateSpline(age, ls, k=1)(range(yN))
        ybar = log(maximum(eff, 1e-15) * maximum(ls, 1e-15))
        zz = linspace(-4 * SIGMAy, 4.5 * SIGMAy, zN)
        zz[-1] = 6 * SIGMAy
        zrange = (zz[1:] + zz[:-1]) / 2.0
        zrange = concatenate(([-float("inf")], zrange, [float("inf")]))

        # labor supply in efficiency unit by age and productivity
        self.ef = zeros((yN, zN))
        for y in range(yN):
            for z in range(zN):
                self.ef[y, z] = exp(ybar[y] + zz[z])

        # transition probability of productivity
        self.pi = zeros((zN, zN))
        for i in range(zN):
            for j in range(zN):
                self.pi[i, j] = (norm.cdf((zrange[j + 1] - GAM * zz[i]) /
                                          SIGMAe, 0, 1) -
                                 norm.cdf((zrange[j] - GAM * zz[i]) /
                                          SIGMAe, 0, 1))

        """distribution of productivity within each age"""
        self.muz = zeros((yN, zN))
        for j in range(zN):
            self.muz[0, j] = (norm.cdf(zrange[j + 1], 0, SIGMAy) -
                              norm.cdf(zrange[j], 0, SIGMAy))
        for y in range(1, yN):
            self.muz[y] = self.muz[y - 1].dot(self.pi)
        self.mu0 = zeros(hN * zN * aN)
        mu0 = self.mu0.reshape(hN, zN, aN)
        mu0[0, :, self.a0_id] = self.muz[0]

    def values(self):
        print('\n===================== Parameters =====================')
        print('Transition over % i periods ... \n' % (self.T))
        print('Newborn population growth rate rises from % 2.2f %%'
              % ((self.npg[0] - 1) * 100),
              'to % 2.2f %%' % ((self.npg[-1] - 1) * 100))
        print('prob. of surviving up to maximum age rises from % 2.2f %%'
              % (prod(self.sp[0]) * 100),
              'to % 2.2f %%' % (prod(self.sp[-1]) * 100))
        print('----------------------preference--------------------', '\n',
              'Weight on housing (PSI): % 2.2f' % (self.PSI), '\n',
              'Time preference  (BET): % 2.3f' % (self.BET), '\n',
              'Inverse of ...  (SIG): % 2.3f' % (self.SIG), '\n',
              '----------------------------Algorithm---------------', '\n',
              'SS_r: % 2.3f;' % (self.SSr),
              'SS_q: % 2.2f;' % (self.SSq),
              'SS_qr: % 2.2f' % (self.SSs), '\n',
              'Tol. % 2.2f %%' % (self.tol * 100),
              'grid scaler: % 2.1f' % (self.gs), '\n',
              'savgol_windows: % i;' % (self.savgol_windows),
              'savgol_order: % i' % (self.savgol_order), '\n',
              '-------------------------technology------------------', '\n',
              'Delta: % i %%' % (self.DEL * 100),
              'Alpha: % 2.2f' % (self.ALP),
              'SIGMAe: % 2.2f' % (self.SIGMAe), '\n',
              'Maximum age: % i;' % (self.yN),
              'Maximum working age: % i' % (self.W), '\n',
              'a_min: % 2.1f' % (self.aa[0]),
              'and a_max: % 2.1f' % (self.aa[-1]),
              'with aN: % i' % (self.aN), '\n',
              'h_min: % 2.1f' % (self.hh[0]),
              'and h_max: % 2.1f' % (self.hh[-1]),
              'with hN: % i' % (self.hN),
              'and hs: % 2.2f' % (self.hs), '\n',
              'SIGMAe: % 2.2f' % (self.SIGMAe),
              'and GAM: % 2.2f' % (self.GAM),
              'with zN: % i' % (self.zN), '\n',
              '--------------------tax, cost and regulation--------', '\n',
              'House Sales Tax: % i %%' % (self.TC * 100), '\n',
              'Tax on liquid asset: % i %%' % (self.Tr * 100), '\n',
              'Tax on labor income: % i %%' % (self.Tw * 100), '\n',
              'Tax on house: % i %%' % (self.Th * 100), '\n',
              'Tax on bequest: % i %%' % (self.ZET * 100), '\n',
              'Replacement ratio (THE): % i %%' % (self.THE * 100), '\n',
              'DTI ratio: % i %%' % (self.dti * 100), '\n',
              'LTV ratio: % i %%' % (self.ltv * 100))
        print('====================================================== \n')


class state:
    """ This class is just a "struct" to hold the collection of primitives defining
    an economy in which one or multiple generations live """

    def __init__(self, par, ss=0, r_0=0.064806, q_0=13.9419, s_0=0.96352,
                 bq_0=0.0554, r_1=0.064806, q_1=13.9419, s_1=0.96352,
                 bq_1=0.0554):
        # tr = 0.429, tw = 0.248, ZET=0.5, gy = 0.195, in Section 9.3.
        # in Heer/Maussner
        """tr, tw and tb are tax rates on capital return, wage and tax
        for pension. tb is determined by replacement ratio, ZET, and
        other endogenous variables.
        gy is ratio of government spending over output. Transfer from
        government to households, Tr, is determined endogenously"""
        def repl(a, T):
            if T == 1:
                return a[0:T]
            else:
                return concatenate((a, repeat(a[-1], T - len(a))))
        # repl = lambda a, T: a[0:T] if T == 1\
        #        else concatenate((a, repeat(a[-1], T - len(a))))
        if ss == 0:
            self.pop = par.pop[0:1]
            self.HS = par.HS[0:1]
        elif ss == 1:
            self.pop = par.pop[-2:-1]
            self.HS = par.HS[-2:-1]
        else:
            self.pop = par.pop
            self.HS = par.HS
        self.T = T = len(self.pop)

        self.transition = (T > 1)
        self.tol = par.tol
        yN, hN, zN, aN = par.yN, par.hN, par.zN, par.aN

        """ Parameters for initial and terminal steady states """
        self.r_0 = r_0
        self.q_0 = q_0
        self.s_0 = s_0
        self.bq_0 = bq_0
        self.r_1 = r_1
        self.q_1 = q_1
        self.s_1 = s_1
        self.bq_1 = bq_1

        """Construct containers for market ps, tax rates, pension, bequest"""
        self.THE = par.THE * ones(T)
        self.Tr = par.Tr * ones(T)
        self.Tw = par.Tw * ones(T)
        self.Th = par.Th * ones(T)

        self.r = repl(linspace(r_0, r_1, 2 * par.yN), T)
        self.q = repl(linspace(q_0, q_1, 2 * par.yN), T)
        self.s = repl(linspace(s_0, s_1, 2 * par.yN), T)
        self.bq = repl(linspace(bq_0, bq_1, 2 * par.yN), T)

        (self.pr, self.L, self.KD, self.w, self.b) = [zeros(T)
                                                      for i in range(5)]
        self.LC = (self.LCa, self.LCh, self.LCc, self.LCr, self.LCC, self.LCR,
                   self.LCA, self.LCH) = [[zeros(yN) for t in range(T)]
                                          for i in range(8)]
        for t in range(T):
            """pr = population of retired agents"""
            self.pr[t] = sum(self.pop[t, par.W:])
            """L = labor supply in efficiency unit"""
            self.L[t] = sum([par.muz[y].dot(par.ef[y]) * self.pop[t, y]
                             for y in range(yN)])
            self.KD[t] = (((self.r[t] + par.DEL) / par.ALP) **
                          (1.0 / (par.ALP - 1.0)) * self.L[t])
            self.w[t] = (((self.r[t] + par.DEL) / par.ALP) **
                         (par.ALP / (par.ALP - 1.0)) * (1.0 - par.ALP))
            self.b[t] = self.THE[t] * self.w[t] * self.L[t] / self.pr[t]

        """ PRICES, PENSION BENEFITS, BEQUESTS AND TAXES that are observed
        by households """
        self.ps = array([self.r, self.w, self.q, self.s, self.b, self.bq,
                         self.THE, self.Tr, self.Tw, self.Th])

    def aggregate(self, par, vmu, vc, vrt, vin):
        """Aggregate Capital, Labor in Efficiency unit and Bequest
        over all cohorts"""
        def my(x):
            if x < self.T:
                return x
            else:
                return -1
        # my = lambda x: x if x < self.T - 1 else -1
        yN, hN, zN, aN = par.yN, par.hN, par.zN, par.aN
        hh, aa = par.hh, par.aa

        self.mu = [array(vmu[t]).reshape(yN, hN, zN, aN)
                   for t in range(len(vmu))]
        self.c = [array(vc[t]).reshape(yN, hN, zN, aN) for t in range(len(vc))]
        self.rt = [array(vrt[t]).reshape(yN, hN, zN, aN)
                   for t in range(len(vrt))]
        self.di = [array(vin[t]).reshape(yN, hN, zN, aN)
                   for t in range(len(vin))]

        self.AG = (self.HD, self.KS, self.BQ, self.PD, self.CO, self.RT,
                   self.DT, self.DI, self.CR) = [zeros(self.T)
                                                 for i in range(9)]
        # for s in self.AG:
        #     s *= 0
        """Aggregate all cohorts' capital and labor supply at each year"""
        for t in range(self.T):
            for y in range(yN):
                """Ks(0) is given, Ks(t) is determined by a(t-1) and E[r(t)]
                in period t-1"""
                ks = (sum(self.mu[my(t + y)][-(y + 1)], (0, 1)).dot(aa) *
                      self.pop[t, -(y + 1)])
                """HD(0) is given, HD(t) is determined by h(t-1) and q(t-1)
                in period t-1"""
                hd = (sum(self.mu[my(t + y)][-(y + 1)], (1, 2)).dot(hh) *
                      self.pop[t, -(y + 1)])
                """Beq(t) are KS(t)+q(t)*H(t) left by those who die
                at the start of period t"""
                bq = ((ks + hd * self.q[t]) * par.spr[t, -(y + 1)] *
                      (1 - par.ZET) / sum(self.pop[t]))
                co = (sum(self.mu[my(t + y)][-(y + 1)] *
                          self.c[my(t + y)][-(y + 1)]) * self.pop[t, -(y + 1)])
                rt = (sum(self.mu[my(t + y)][-(y + 1)] *
                          self.rt[my(t + y)][-(y + 1)]) *
                      self.pop[t, -(y + 1)])
                cr = (sum(self.mu[my(t + y)][-(y + 1)] *
                          maximum(self.rt[my(t + y)][-(y + 1)], 0)) *
                      self.pop[t, -(y + 1)])
                di = (sum(self.mu[my(t + y)][-(y + 1)] *
                          self.di[my(t + y)][-(y + 1)]) *
                      self.pop[t, -(y + 1)])
                dt = (sum(self.mu[my(t + y)][-(y + 1)], (0, 1)).
                      dot(minimum(aa, 0)) * self.pop[t, -(y + 1)])
                self.KS[t] += ks
                self.HD[t] += hd
                self.BQ[t] += bq
                self.CO[t] += co
                self.RT[t] += rt
                self.DT[t] += dt
                self.DI[t] += di
                self.CR[t] += cr
        self.PD = self.KS**par.ALP * self.L**(1.0 - par.ALP)

    def stat(self, t=0):
        print('=================Summary Statistics===================')
        print("DT/DI:%2.1f %%" % (-self.DT[t] / self.DI[t] * 100))
        print("DT/Y:%2.1f %%" % (-self.DT[t] / self.PD[t] * 100))
        print("C/Y:%2.1f %%" % (self.CO[t] / self.PD[t] * 100))
        print("(C+R)/Y:%2.1f %%"
              % ((self.CO[t] + self.s[t] * self.CR[t]) / self.PD[t] * 100))
        print("(DI-C)/DI:%2.1f %%"
              % ((self.DI[t] - self.CO[t]) / self.DI[t] * 100))
        print("(DI-C-R)/DI:%2.1f %%"
              % ((self.DI[t] - self.CO[t] - self.s[t] * self.CR[t]) /
                 self.DI[t] * 100))
        print("(KD+qH)/Y:%2.1f %%"
              % ((self.KS[t] + self.q[t] * self.HD[t]) / self.PD[t] * 100))
        print("KD/Y:%2.1f %%" % (self.KS[t] / self.PD[t] * 100))
        print("s/q:%2.1f %%" % (self.s[t] / self.q[t] * 100))
        print("KD/L:%2.2f" % (self.KS[t] / self.L[t]))
        print('======================================================')

    def update(self, par, n=0):
        """ Update market ps, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,T from last
        iteration """
        K0 = self.KD  # capital demand that is consistent with interest rate
        rs = par.ALP * (self.KS / self.L)**(par.ALP - 1.0) - par.DEL
        self.bq = self.BQ
        if self.transition:
            self.r[1:] = self.r[1:] * (1 + par.SSr *
                                       (self.KD[1:] - self.KS[1:]) * 2 /
                                       (self.KD[1:] + self.KS[1:]))
            self.q[:-1] = self.q[:-1] * (1 + par.SSq *
                                         (self.HD[1:] - self.HS[1:]) /
                                         self.HS[1:])
            self.s = self.s * (1 + par.SSs * self.RT / self.HS)
        else:
            self.r = self.r * (1 + random() * par.SSr * (self.KD - self.KS) *
                               2 / (self.KD + self.KS))
            self.q = self.q * (1 + random() * par.SSq * (self.HD - self.HS) / self.HS)
            self.s = self.s*(1+random()*par.SSs*self.RT/self.HS)
        self.KD = ((self.r+par.DEL)/par.ALP)**(1.0/(par.ALP-1.0))*self.L
        self.w = ((self.r+par.DEL)/par.ALP)**(par.ALP/(par.ALP-1.0))*(1.0-par.ALP)
        self.b = self.THE*self.w*self.L/self.pr
        if self.transition:
            time = datetime.now()
            r0 = self.r; q0 = self.q; qr0 = self.s
            if par.savgol_windows > 1:
                self.r[1:] = savgol_filter(r0[1:], par.savgol_windows, par.savgol_order)
                self.q = savgol_filter(q0, par.savgol_windows, par.savgol_order)
                self.s = savgol_filter(qr0, par.savgol_windows, par.savgol_order)
            rmin = min(self.r_0,self.r_1) - 0.02
            rmax = max(self.r_0,self.r_1) + 0.02
            qmin = min(self.q_0,self.q_1) - 4.0
            qmax = max(self.q_0,self.q_1) + 4.0
            qrmin = min(self.s_0,self.s_1) - 0.10
            qrmax = max(self.s_0,self.s_1) + 0.10
            title = "TP at d%i"%(time.day) + "h%i"%(time.hour) + " after %i loops"%(n)
            filename = title + '.png'
            fig = plt.figure(facecolor='white')
            plt.rcParams.update({'font.size': 8})
            ax = fig.add_subplot(111)
            ax1 = fig.add_subplot(331)
            ax2 = fig.add_subplot(332)
            ax3 = fig.add_subplot(334)
            ax4 = fig.add_subplot(335)
            ax7 = fig.add_subplot(333)
            ax8 = fig.add_subplot(336)
            ax5 = fig.add_subplot(337)
            ax6 = fig.add_subplot(338)
            fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None,
                                    top=None, bottom=None)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                            right='off')
            ax1.plot(self.KS/sum(self.pop,axis=1),'.',label='supply')
            ax1.plot(K0/sum(self.pop,axis=1),'.',label='demand')
            ax2.plot(self.HS/sum(self.pop,axis=1),'.',label='supply')
            ax2.plot(self.HD/sum(self.pop,axis=1),'.',label='demand')
            ax3.plot(0,self.r_0,'o',label='initial')
            ax3.plot(r0,'.',label='updated')
            ax3.plot(self.r,'.',label='smoothed')
            ax3.plot(rs,'.',label='implied')
            ax4.plot(0,self.q_0,'o',label='initial')
            ax4.plot(q0,'.',label='updated')
            ax4.plot(self.q,'.',label='smoothed')
            ax5.plot(self.PD/sum(self.pop,axis=1),'.',label='output')
            ax5.plot(self.CO/sum(self.pop,axis=1),'.',label='consumption')
            ax5.plot((self.CO+self.s*self.CR)/sum(self.pop,axis=1),'.',label='expenditure')
            ax6.plot((self.DI-self.CO)/self.DI,'.',label='saving rate')
            ax6.plot((self.DI-self.CO-self.s*self.CR)/self.DI,'.',label='saving rate with rent')
            ax7.plot((self.RT)/self.HS*100,'.',label='demand')
            ax8.plot(0,self.s_0,'o',label='initial')
            ax8.plot(qr0,'.',label='updated')
            ax8.plot(self.s,'.',label='smoothed')
            ax1.legend(prop={'size':7})
            ax2.legend(prop={'size':7})
            ax3.legend(prop={'size':7})
            ax4.legend(prop={'size':7}, loc=4)
            # ax5.legend(prop={'size':7})
            ax6.legend(prop={'size':7})
            ax7.legend(prop={'size':7})
            ax8.legend(prop={'size':7})
            # ax6.legend(prop={'size':7})
            ax1.axis([0, self.T, 1.05, 1.18])
            ax2.axis([0, self.T, 0.16, 0.26])
            ax3.axis([0, self.T, rmin, rmax])
            ax4.axis([0, self.T, qmin, qmax])
            # ax5.axis([0, self.T, 0.2, 0.8])
            ax6.axis([0, self.T, 0.28, 0.42])
            ax7.axis([0, self.T, -10, 10])
            ax8.axis([0, self.T, qrmin, qrmax])
            ax.set_title('Transition over %i periods'%(self.T), y=1.08)
            ax1.set_title('Liquid Asset')
            ax2.set_title('House')
            ax3.set_title('Interest Rate')
            ax4.set_title('House Price')
            ax5.set_title('Production and Consumption')
            ax6.set_title('Saving Rate')
            ax7.set_title('RT')
            ax8.set_title('Rental Price')
            if system() == 'Windows':
                path = 'D:\Huggett\Figs'
            else:
                path = '/Users/hyunchangyi/GitHub/Huggett/Figs'
            fullpath = os.path.join(path, filename)
            fig.savefig(fullpath, dpi=300)
            plt.close()
        self.ps[0] = self.r
        self.ps[1] = self.w
        self.ps[2] = self.q
        self.ps[3] = self.s
        self.ps[4] = self.b
        self.ps[5] = self.bq

    def converged(self):
        return max(absolute(self.KD - self.KS))/max(self.KD) < self.tol \
                and max(absolute(self.HD - self.HS))/max(self.HS) < self.tol \
                and max(absolute(self.RT)/max(self.HS)) < self.tol

    def prices(self,n=0,t=0):
        print("n=%i"%(n),"t=%i"%(t),"r=%2.4f%%"%(self.r[t]*100),\
              "Kd=%3.2f%%,"%((self.KD[t]-self.KS[t])*2/(self.KD[t]+self.KS[t])*100),\
              "q=%2.4f,"%(self.q[t]),\
              "HD=%3.2f%%,"%((self.HD[t]-self.HS[t])/self.HS[t]*100),\
              "s=%1.5f,"%((self.s[t])),\
              "Rd=%3.2f%%,"%((self.RT[t])/self.HS[t]*100),\
              "bq=%2.4f," %(self.BQ[t]),\
              "b=%2.4f," %(self.b[t]))

    def plot(self, par, t=0, yi=15, yt=70, ny=4):
        """plot life-path of aggregate capital accumulation and house demand"""
        yN, hN, zN, aN = par.yN, par.hN, par.zN, par.aN
        hh, aa = par.hh, par.aa
        pop = par.pop

        mu = self.mu[t]
        c = self.c[t]
        rt = self.rt[t]
        di = self.di[t]
        lz_a = zeros(aN)
        lz_h = zeros(hN)
        lz_ah = zeros((hN,aN))
        mupop = zeros((yN,hN,zN,aN))
        """Aggregate all cohorts' capital and labor supply at each year"""
        for y in range(yN):
            self.LCA[t][y] = sum(mu[y],(0,1)).dot(aa)*pop[t,y]
            self.LCH[t][y] = sum(mu[y],(1,2)).dot(hh)*pop[t,y]
            self.LCa[t][y] = sum(mu[y],(0,1)).dot(aa)
            self.LCh[t][y] = sum(mu[y],(1,2)).dot(hh)
            self.LCC[t][y] = sum(mu[y]*c[y],(0,1,2))*pop[t,y]
            self.LCR[t][y] = sum(mu[y]*rt[y],(0,1,2))*pop[t,y]
            self.LCc[t][y] = sum(mu[y]*c[y],(0,1,2))
            self.LCr[t][y] = sum(mu[y]*rt[y],(0,1,2))
            lz_a += sum(mu[y],(0,1))*pop[t,y]
            lz_h += sum(mu[y],(1,2))*pop[t,y]
            """ ah: hN by aN matrix that represents populations of each pairs
            of house and asset holders """
            lz_ah += sum(mu[y],1)*pop[t,y]
            mupop[y] = mu[y]*pop[t,y]
        w = hh[:,newaxis]*self.q[t] + aa[newaxis,:]
        unsorted = array((lz_ah.ravel(),w.ravel())).T
        lz_ah, w = unsorted[unsorted[:,1].argsort()].T
        unsorted = array((mupop.ravel(),di.ravel())).T
        mupop, di = unsorted[unsorted[:,1].argsort()].T
        title = 'aN=%3i'%(aN) + ' hN=%2i'%(hN) + ' r=%2.2f%%'%(self.r[t]*100) \
               + ' q=%2.2f'%(self.q[t]) + ' s=%2.2%%f'%(self.s[t]*100)
        if self.transition:
            title = 'In Trans., at %i '%(t) + title
        else:
            title = 'In SS, ' + title
        filename = title + '.png'
        fig = plt.figure(facecolor='white')
        plt.rcParams.update({'font.size': 8})
        # matplotlib.rcParams.update({'font.size': 22})
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(331)
        ax2 = fig.add_subplot(334)
        ax3 = fig.add_subplot(332)
        ax4 = fig.add_subplot(335)
        ax5 = fig.add_subplot(333)
        ax6 = fig.add_subplot(336)
        ax7 = fig.add_subplot(337)
        ax8 = fig.add_subplot(338)
        ax9 = fig.add_subplot(339)
        fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, \
                                                       top=None, bottom=None)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', \
                                                       left='off', right='off')
        ax1.plot(range(yN),self.LCA[t],label='aggregate')
        ax1.plot(range(yN),self.LCa[t],label='per capita')
        ax2.plot(range(yN),self.LCH[t],label='aggregate')
        ax2.plot(range(yN),self.LCh[t],label='per capita')
        ax7.plot(range(yN),self.LCC[t],label='aggregate')
        ax7.plot(range(yN),self.LCc[t],label='per capita')
        ax8.plot(range(yN),self.LCR[t],label='aggregate')
        ax8.plot(range(yN),self.LCr[t],label='per capita')
        for y in linspace(yi,yt,ny).astype(int):
            ax3.plot(aa,sum(mu[y],(0,1)),label='age %i'%(y))
        for y in linspace(yi,yt,ny).astype(int):
            ax4.plot(hh,sum(mu[y],(1,2)),label='age %i'%(y))
        lx = cumsum(lz_a)/sum(lz_a)
        lz_aa = lz_a*aa - min(lz_a*aa)
        ly = cumsum(lz_aa)/sum(lz_aa)
        gini_l = 1 - sum([(ly[i]+ly[i-1])*(lx[i]-lx[i-1]) for i in range(1,len(lx))])
        ax5.plot(lx,ly,".",label='Gini coef.=%1.2f'%(gini_l))
        lx = cumsum(lz_ah)/sum(lz_ah)
        lz_ahw = lz_ah*w - min(lz_ah*w)
        ly = cumsum(lz_ahw)/sum(lz_ahw)
        gini_t = 1 - sum([(ly[i]+ly[i-1])*(lx[i]-lx[i-1]) for i in range(1,len(lx))])
        ax6.plot(lx,ly,".",label='Gini coef.=%1.2f'%(gini_t))
        # ax1.legend(bbox_to_anchor=(0.9,1.0),loc='center',prop={'size':8})
        lx = cumsum(mupop)/sum(mupop)
        ly = cumsum(mupop*di)/sum(mupop*di)
        gini_i = 1 - sum([(ly[i]+ly[i-1])*(lx[i]-lx[i-1]) for i in range(1,len(lx))])
        ax9.plot(lx,ly,".",label='Gini coef.=%1.2f'%(gini_i))
        ax1.legend(prop={'size':7}, loc=2)
        ax2.legend(prop={'size':7}, loc=2)
        ax3.legend(prop={'size':7})
        ax4.legend(prop={'size':7})
        ax5.legend(prop={'size':7}, loc=2)
        ax6.legend(prop={'size':7}, loc=2)
        ax7.legend(prop={'size':7}, loc=2)
        ax8.legend(prop={'size':7}, loc=2)
        ax9.legend(prop={'size':7}, loc=2)
        # ax3.axis([0, 15, 0, 0.1])
        ax5.axis([0, 1, 0, 1])
        ax6.axis([0, 1, 0, 1])
        ax9.axis([0, 1, 0, 1])
        # ax4.axis([0, 80, 0, 1.0])
        ax1.set_xlabel('Age')
        ax2.set_xlabel('Age')
        ax7.set_xlabel('Age')
        ax8.set_xlabel('Age')
        ax3.set_xlabel('Asset Size')
        ax4.set_xlabel('House Size')
        # ax5.set_xlabel('Cum. Share of Agents from Lower to Higher')
        ax5.set_xlabel('Cum. Share of Agents')
        ax6.set_xlabel('Cum. Share of Agents')
        ax9.set_xlabel('Cum. Share of Agents')
        # ax5.set_ylabel('Cum. Share of Asset Occupied')
        # ax6.set_ylabel('Cum. Share of House Occupied')
        # ax6.set_ylabel('Cum. Share of Total Wealth')
        # ax9.set_ylabel('Cum. Share of Disposable Income')
        ax.set_title(title, y=1.08)
        ax1.set_title('Life-Cycle Liquid Asset')
        ax2.set_title('Life-Cycle House')
        ax7.set_title('Life-Cycle Consumption')
        ax8.set_title('Life-Cycle RT')
        ax3.set_title('Dist. of Liquid Asset')
        ax4.set_title('Dist. of House Size')
        ax5.set_title('Lorenz Curve for Liquid Asset')
        # ax6.set_title('Lorenz Curve for House')
        ax6.set_title('Lorenz Curve for Total Wealth')
        ax9.set_title('Lorenz Curve for Disposable Income')
        if system() == 'Windows':
            path = 'D:\Huggett\Figs'
        else:
            path = '/Users/hyunchangyi/GitHub/Huggett/Figs'
        fullpath = os.path.join(path, filename)
        fig.savefig(fullpath, dpi=300)
        # fig.savefig(fullpath, dpi=600)
        # ax4.axis([0, 80, 0, 1.1])
        # plt.show()
        plt.close()


class cohort:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, par, ss=0, sy=0):
        # self.yN = yN = (y+1 if (y >= 0) and (y <= W+R-2) else W+R)
        yN, hN, zN, aN = par.yN, par.hN, par.zN, par.aN
        self.sy = sy
        self.sp = par.sp[0] if ss == 0 else par.sp[-1]
        # agents start their life with asset aa[a0_id]
        self.a0_id = par.a0_id
        """ container for value function and expected value function """
        # v[y,h,j,i] is the value of y-yrs-old agent with asset i and prod. j, house h
        self.v = zeros((yN,hN,zN,aN))
        """ container for policy functions,
        which are used to calculate vmu and not stored """
        self.a = zeros((yN,hN,zN,aN)).astype(int)
        self.h = zeros((yN,hN,zN,aN)).astype(int)
        # self.c = zeros((yN,hN,zN,aN))
        """ distribution of agents w.r.t. age, productivity and asset
        for each age, distribution over all productivities and assets add up to 1 """
        self.vmu = zeros(yN*hN*zN*aN)
        self.vc = zeros(yN*hN*zN*aN)
        self.vrt = zeros(yN*hN*zN*aN)
        self.vin = zeros(yN*hN*zN*aN)
        self.mu_sy = par.mu0


    def optimalpolicy(self,par,ps):
        """ Given ps, transfers, benefits and tax rates over one's life-cycle,
        value and decision functions are calculated ***BACKWARD*** """
        yN, hN, zN, aN = par.yN, par.hN, par.zN, par.aN
        hh, aa = par.hh, par.aa
        ps = concatenate((repeat(ps[:,0][:,newaxis],max(yN-ps.shape[1],0),axis=1),ps),axis=1)
        r, w, q, s, b, bq, THE, Tr, Tw, Th = ps
        sp = self.sp
        sy = self.sy

        """ev[y,nh,j,ni] is the expected value when next period asset ni and house hi"""
        ev = zeros((yN,hN,zN,aN))
        """ei[y,nh,z,ni] is the expected value of lifetime incomes
        when current productivity z and next period asset ni and house hi"""
        ei = zeros((yN,hN,zN,aN))
        """ct is a channel to store optimal consumption in vc"""
        ct = self.vc.reshape(yN,hN,zN,aN)
        rtt = self.vrt.reshape(yN,hN,zN,aN)
        inc = self.vin.reshape(yN,hN,zN,aN)
        """ inline functions: utility and income adjustment by trading house """
        """!!!!!!!!!!!!!!!!!!  from here  !!!!!!!!!!!!!!!!!!!"""
        util = lambda c, r, h: ((maximum(c,1e-15)**(1-par.PSI)*maximum(h+r,1e-15)**par.PSI)\
                                    **(1-par.SIG))/(1-par.SIG)
        """!!!!!!!!!!!!!!!!!!  to here  !!!!!!!!!!!!!!!!!!!!!"""
        hinc = lambda h, nh, q: (h-nh)*q - par.TC*h*q*(h!=nh)
        """ y = -1 : just before the agent dies """
        for h in range(hN):
            for z in range(zN):
                B = aa*(1+(1-Tr[-1]*(aa>0))*r[-1]) + hinc(hh[h],hh[0],q[-1]) \
                        - hh[h]*q[-1]*Th[-1] \
                        + w[-1]*par.ef[-1,z]*(1-THE[-1]-Tw[-1]) + b[-1] + bq[-1]
                ct[-1,h,z] = (1-par.PSI)*(B + s[-1]*hh[h])
                """!!!!!!!!!!!!!!!!!!  from here  !!!!!!!!!!!!!!!!!!!"""
                rtt[-1,h,z] = (B-ct[-1,h,z])/s[-1]
                self.v[-1,h,z] = util(ct[-1,h,z],rtt[-1,h,z],hh[h])
                cv = (ct[-1,h,z] <= 0.0) | (rtt[-1,h,z]+hh[h] <= 0.0)
                self.v[-1,h,z][cv] = -float("inf")
                """!!!!!!!!!!!!!!!!!!  to here  !!!!!!!!!!!!!!!!!!!!!"""
                inc[-1,h,z] = - s[-1]*minimum(rtt[-1,h,z],0) - Th[-1]*q[-1]*hh[h] \
                                + b[-1] + bq[-1] + aa*(1-Tr[-1]*(aa>0))*r[-1]
            ev[-1,h] = par.pi.dot(self.v[-1,h])
            ei[-1,h] = par.pi.dot(inc[-1,h])
        """ y = -2, -3,..., -60 """
        for y in range(-2, sy-yN-1, -1):
            for h in range(hN):
                for z in range(zN):
                    vt = zeros((hN,aN,aN))
                    for nh in range(hN):
                        p = aa*(1+(1-Tr[y]*(aa>0))*r[y]) + b[y]*(y>=-par.R) \
                                + w[y]*par.ef[y,z]*(1-THE[y]-Tw[y]) + bq[y] \
                                + hinc(hh[h],hh[nh],q[y]) - Th[y]*q[y]*hh[h]
                        B = p[:,newaxis] - aa
                        """!!!!!!!!!!!!!!!!!!  from here  !!!!!!!!!!!!!!!!!!!"""
                        c = (1-par.PSI)*(B + s[y]*hh[h])
                        rt = (B-c)/s[y]
                        vt[nh] = util(c,rt,hh[h]) + par.BET*sp[y+1]*ev[y+1,nh,z]
                        cv = (c<=0.0)|(rt+hh[h]<=0.0)|(aa+minimum(par.ltv*hh[nh]*q[y+1],
                                        par.dti*par.pi[z].dot(ei[y+1,nh])/(1+r[y+1])) < 0.0)
                        vt[nh][cv] = -float("inf")
                        """!!!!!!!!!!!!!!!!!!  to here  !!!!!!!!!!!!!!!!!!!!!"""
                    for a in range(aN):
                        """find optimal pairs of house and asset """
                        self.h[y,h,z,a], self.a[y,h,z,a] \
                            = unravel_index(vt[:,a,:].argmax(),vt[:,a,:].shape)
                        self.v[y,h,z,a] = vt[self.h[y,h,z,a],a,self.a[y,h,z,a]]
                        B = aa[a]*(1+(1-Tr[y]*(aa[a]>0))*r[y]) + b[y]*(y>=-par.R) \
                                        + bq[y] + w[y]*par.ef[y,z]*(1-THE[y]-Tw[y]) \
                                        + hinc(hh[h],hh[self.h[y,h,z,a]],q[y]) \
                                        - aa[self.a[y,h,z,a]] - Th[y]*q[y]*hh[h]
                        ct[y,h,z,a] = (1-par.PSI)*(B+s[y]*hh[h])
                        rtt[y,h,z,a] = (B-ct[y,h,z,a])/s[y]
                        inc[y,h,z,a] = - s[y]*minimum(rtt[y,h,z,a],0) + b[y]*(y>=-par.R) \
                                        + bq[y] + aa[a]*(1-Tr[y]*(aa[a]>0))*r[y] \
                                        + w[y]*par.ef[y,z]*(1-THE[y]-Tw[y]) \
                                        - Th[y]*q[y]*hh[h]
                        ei[y,h,z,a] = inc[y,h,z,a] + par.pi[z].dot(ei[y+1,
                                        self.h[y,h,z,a],:,self.a[y,h,z,a]])/(1+r[y+1])
                ev[y,h] = par.pi.dot(self.v[y,h])
        """ find distribution of agents w.r.t. age, productivity and asset """
        self.vmu *= 0
        mu = self.vmu.reshape(yN,hN,zN,aN)
        mu[sy] = self.mu_sy.reshape(hN,zN,aN)
        for y in range(sy+1,yN):
            for h in range(hN):
                for z in range(zN):
                    for a in range(aN):
                        mu[y,self.h[y-1,h,z,a],:,self.a[y-1,h,z,a]] += mu[y-1,h,z,a]*par.pi[z]


"""The following are procedures to get steady state of the economy using direct
age-profile iteration and projection method"""

def fss(par, k, c, N=300):
    """Find Old and New Steady States with population growth rates ng and ng1"""
    start_time = datetime.now()
    for n in range(N):
        c.optimalpolicy(par,k.ps)
        k.aggregate(par,[c.vmu],[c.vc],[c.vrt],[c.vin])
        k.prices(n=n+1)
        k.update(par,n=n+1)
        if k.converged():
            print('Converged.')
            break
        if n >= N-1:
            print('Not converged.')
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return k, c


# separate the procedure of finding optimal policy of each cohort for Parallel Process

def sub1(t,vmu,vc,vrt,vin,ps,c0,c1,par):
    """Consider a generation who are at their maximum age in period t.
    sy is their age in period 0"""
    T, yN, hN, zN, aN = par.T, par.yN, par.hN, par.zN, par.aN
    sy = max((yN-1)-t,0)
    c = cohort(par,ss=2,sy=sy)
    if sy > 0:  c.mu_sy = c0.vmu.reshape(yN,hN*zN*aN)[sy]
    if t < T-1:
        c.optimalpolicy(par,ps[:,max(t-yN+1,0):t+1])
    else:
        c.vmu, c.vc, c.vrt, c.vin = c1.vmu, c1.vc, c1.vrt, c1.vin
    for i in range(yN*hN*zN*aN):
        vmu[i] = c.vmu[i]
        vc[i] = c.vc[i]
        vrt[i] = c.vrt[i]
        vin[i] = c.vin[i]


def tran(par,kt,c0,c1,N=5):
    T = par.T
    vl = par.yN*par.hN*par.zN*par.aN
    """Generate mu of T cohorts who die in t = 0,...,T-1 with initial asset g0.apath[-t-1]"""
    VM = [RawArray(c_double, vl) for t in range(T)]
    VC = [RawArray(c_double, vl) for t in range(T)]
    VRT = [RawArray(c_double, vl) for t in range(T)]
    VIN = [RawArray(c_double, vl) for t in range(T)]
    for n in range(N):
        start_time = datetime.now()
        print(str(n+1)+'th loop started at {}'.format(start_time))
        jobs = []
        # for t, vmu in enumerate(VM):
        for t in range(T):
            p = Process(target=sub1, args=(t,VM[t],VC[t],VRT[t],VIN[t],kt.ps,c0,c1,par))
            p.start()
            jobs.append(p)
            # if t % 40 == 0:
            #     print 'processing another 40 cohorts...'
            if len(jobs) % 8 == 0:
                for p in jobs:
                    p.join()
                    jobs=[]
        if len(jobs) > 0:
            for p in jobs:
                p.join()
        kt.aggregate(par,VM,VC,VRT,VIN)
        for t in linspace(1,T-1,20).astype(int):
            kt.prices(n=n+1,t=t)
        kt.update(par,n=n+1)
        end_time = datetime.now()
        print('this loop finished in {}\n'.format(end_time - start_time))
        if kt.converged():
            print('Transition Path Converged! in', n+1,'iterations.')
            break
        if n >= N-1:
            print('Transition Path Not Converged! in', n+1,'iterations.')
            break
    return kt, VM, VC, VRT, VIN


def find_trans(contd=0, SSr=0.015, SSq=0.05, SSs=0.05, T=240, windows=3, N=5):
    start_time = datetime.now()
    if contd == 0:
        with open('ss0.pickle','rb') as f:
            [k0, c0, par0] = pickle.load(f)
        with open('ss1.pickle','rb') as f:
            [k1, c1, par] = pickle.load(f)
        kt = state(par, ss=2, r_0=k0.r, r_1=k1.r, q_0=k0.q, q_1=k1.q,
                    s_0=k0.s, s_1=k1.s, bq_0=k0.bq, bq_1=k1.bq)
    else:
        with open('transition.pickle','rb') as f:
            [kt, par] = pickle.load(f)        
        with open('ss0.pickle','rb') as f:
            [k0, c0, par0] = pickle.load(f)
        with open('ss1.pickle','rb') as f:
            [k1, c1, par1] = pickle.load(f)
    par.SSr = SSr
    par.SSq = SSq
    par.SSs = SSs
    par.savgol_windows = windows
    kt, mu, vc, vrt, vin = tran(par, kt, c0, c1, N=N)
    # for t in linspace(0,par.T-1,4).astype(int):
    #     kt.plot(par,t=t,yi=10,ny=5)
    with open('transition.pickle','wb') as f:
        pickle.dump([kt, par], f)
    end_time = datetime.now()
    print('Total Time: {}'.format(end_time - start_time))


if __name__ == '__main__':
    par = params(T=200, aN=60, aL=-10, aH=50, hL=0.0, hH=1.5, hs=0.2, hN=5, zN=18,
            ALP=0.32, PSI=0.4, DEL=0.08, BET=0.998, SIG=2.0,
            Tr=0.10, Tw=0.15, Th=0.01, THE=0.1, ZET=0.15, TC=0.02,
            dti=7/10.0, ltv=5/10.0, spm=0.1,
            savgol_windows=51, savgol_order=1,
            SSr=0.12, SSq=0.12, SSs=0.09, gs=2.0, tol=0.001)
    par.values()

    k0 = state(par,ss=0,r_0=0.064806,q_0=13.9419,s_0=0.96352,bq_0=0.0554)
    c0 = cohort(par,ss=0)
    k0, c0 = fss(par, k0, c0, N=300)
    # k0.plot(par)
    # k0.stat()
    # with open('ss0.pickle','wb') as f:
    #     pickle.dump([k0, c0, par], f)

    # k1 = state(par,ss=1,r_0=0.059326,q_0=13.9120,s_0=0.89889,bq_0=0.0678)
    # c1 = cohort(par,ss=1)
    # k1, c1 = fss(par, k1, c1, N=300)
    # k1.plot(par)
    # k1.stat()
    # with open('ss1.pickle','wb') as f:
    #     pickle.dump([k1, c1, par], f)

    # find_trans(contd=0, SSr=0.09, SSq=0.12, SSs=0.12, T=200, windows=15, N=25)
    # find_trans(contd=1, SSr=0.09, SSq=0.12, SSs=0.12, T=200, windows=7, N=10)
    # find_trans(contd=1, SSr=0.12, SSq=0.07, SSs=0.07, T=180, windows=5, N=14)
    # find_trans(contd=1, SSr=0.03, SSq=0.03, SSs=0.03, T=180, windows=1, N=10)
    # find_trans(contd=1, SSr=0.02, SSq=0.01, SSs=0.01, T=180, windows=1, N=1)
    # find_trans(contd=1, SSr=0.12, SSq=0.07, SSs=0.07, T=240, windows=31, N=10)


    # with open('transition.pickle','rb') as f:
    #     [kt, par] = pickle.load(f)
    # print kt.KD[0:10]/sum(kt.pop[0:10],axis=1)
    # print kt.r[0:10]
    # print kt.q[0:10]
    # print kt.PD[0:10]/sum(kt.pop[0:10],axis=1)
    # print kt.CO[0:10]/sum(kt.pop[0:10],axis=1)
    # print kt.DI[0:10]/sum(kt.pop[0:10],axis=1)
    # print (kt.DI[0:10]-kt.CO[0:10])/kt.DI[0:10]
