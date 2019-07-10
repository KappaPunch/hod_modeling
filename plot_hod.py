# This script uses variables without little-h, but halomod uses units with little-h (Mpc versus Mpc/h etc.), so there are conversions throughout

## ==STANDARD MODULES==
import random
import time, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import\
    InterpolatedUnivariateSpline as spline
from astropy.cosmology import default_cosmology

## ==HALO MODEL & FITTING MODULES==
import halomod as hm
from halomod.integrate_corr import angular_corr_gal

# sections in main() function
# (1) [_MON]: load in tables
# (2) [_TUE]: initialize halo model using halomod
# (3) [_WED]: build probability functions for running mcmc
# (4) [_THU]: generate inital points from random
# (5) [_FRI]: MAIN PART for mcmc process
# (6) [_SAT]: based on mcmc samples, derive some parameters like effective bias
# (7) [_SUN]: save results to files

if __name__ == '__main__':

    def GenerateNz(zmean, sigma, zmin, zmax, flag):
        znum = 100
        z = np.linspace(zmin, zmax, znum)

        if flag == 'Flat':
            nz = np.ones(znum)
        if flag == 'Gaussian':
            rv = norm(loc=zmean, scale=sigma)
            nz = rv.pdf(z)

        return spline(z, nz)

    def GetParams():
    
        params = InitializeParameters()
    
        filename = GetParamFilename()
        fileParams = np.loadtxt(filename, dtype=np.str, usecols=(0,1))
        fileParamsDict = {}
        for idx in range(fileParams.shape[0]):
            fileParamsDict[fileParams[idx,0]] = fileParams[idx,1]
    
        diffFromInit = set(params.keys()).difference(fileParamsDict.keys())
        diffFromInit = list(diffFromInit)
        stdout('There are %d parameters not found in .param file' %
            len(diffFromInit)
        )
    
        for pname, pvalue in params.items():
            if pname in diffFromInit:
                pass
            else:
                params[pname] = fileParamsDict[pname].astype(type(pvalue))
    
        return params
    
    def GetParamFilename():
    
        if len(sys.argv) == 2:
            stdout('read parameter file: %s ' % sys.argv[1])
            return sys.argv[1]
        else:
            sys.exit(
                '%s... please input a parameter file by using command : \
                $python halo_modeling [YOUR_PARAMETER_FILES]' % sys.argv[0]
            )
    
    def InitializeParameters():
    
        return {
                # FILES
                'WORKING_DIRECTORY' : 'YOUR_WORKING_DIRECTORY' ,
                'INPUT_ACF'         : 'FILE_NAME_PLS'          ,
                'INPUT_ZD'          : 'FILE_NAME_PLS'          ,
                'INPUT_COV'         : 'FILE_NAME_PLS'          ,
                'OUTPUT_ACF_PLOT'   : 'FILE_NAME_PLS'          ,
                'OUTPUT_ACF_PTS'    : 'FILE_NAME_PLS'          ,
                'OUTPUT_HON_PLOT'   : 'FILE_NAME_PLS'          ,
                'OUTPUT_HON_PTS'    : 'FILE_NAME_PLS'          ,
                # MODELS
                'VERSION'                       : 'v1'       ,
                'COSMOLOGY'                     : 'Planck15' ,
                'HOD_MODEL'                     : 'Zheng05'  , 
                'HALO_MASS_FUNCTION'            : 'Tinker10' ,
                'HALO_BIAS_FUNCTION'            : 'Tinker10' ,
                'CONCENTRAION_TO_MASS_RELATION' : 'Duffy08'  ,
                'REDSHIFT_DISTR'                : 'Flat'     ,
                # SWITCHES
                'PTS_ONLY'                 : False ,
                'MODEL_ONLY'               : True  ,
                'APPLY_INTEGRAL_CONSTRAIN' : True  ,
                'LITTLE_H_INCUSION'        : True  ,
                # HOD settings
                'obs_number_density' : 0.0005      ,
                'err_obs_ndens'      : 0.00002     ,
                'z_mean'             : 1.12        ,
                'z_sig'              : 0.45        ,
                'z_min'              : 1.0         ,
                'z_max'              : 1.25        ,
                'z_num'              : 100         ,
                'logM_min'           : 6.          ,
                'logM_max'           : 16.         ,
                'theta_min'          : 1./3600.    ,
                'theta_max'          : 3600./3600. ,
                'theta_num'          : 60          ,
                'rmin'               : -5.         ,
                'rmax'               : 100.        ,
                'rnum'               : 500.        ,
                'logu_min'           : -3.         ,
                'logu_max'           : 2.5         ,
                'unum'               : 150         ,
                'log_Mmin'           : 12.2        , # this term will be double-used
                'log_Msat'           : 14.         ,
                'log_Mcut'           : 9.          ,
                'alpha'              : 0.7         , # this term will be double-used
                'sigma'              : 0.25        , # this term will be double-used
                'log_Mc'             : 11.6222     ,
                'Fca'                : 0.5         ,
                'Fcb'                : 0.          ,
                'Fs'                 : 1.          ,
                'delta'              : 1.          ,
                'x'                  : 1.          ,
               }
    
    def stdout(message):
        print('%s...  %s' % (sys.argv[0], message))

    paramDict = GetParams()

    version = paramDict['VERSION']
    stdout("Version : %s" % version)

    wd               = paramDict['WORKING_DIRECTORY']
    corrFilename     = paramDict['INPUT_ACF']
    rdFilename       = paramDict['INPUT_ZD']
    covFilename      = paramDict['INPUT_COV']
    acfModelFilename = paramDict['OUTPUT_ACF_PTS']
    acfModelPlotname = paramDict['OUTPUT_ACF_PLOT']
    honModelFilename = paramDict['OUTPUT_HON_PTS']
    honModelPlotname = paramDict['OUTPUT_HON_PLOT']

    if not paramDict['MODEL_ONLY']:
        data = np.genfromtxt(wd+corrFilename)
        obsSep = data[:,0]
        obsACF = data[:,1]
        obsRR  = data[:,2]

        cov = np.loadtxt(wd+covFilename)
        obsACFerr = np.sqrt(cov.diagonal())
        invCov = np.linalg.inv(cov)

    if paramDict['REDSHIFT_DISTR'] == 'User':
        nz = np.loadtxt(wd+rdFilename)
        redshift_distribution = spline(nz[:,0], nz[:,1])
    else:
        print 'ohyeah'
        redshift_distribution = GenerateNz(paramDict['z_mean'],
                                           paramDict['z_sig'] ,
                                           paramDict['z_min'] ,
                                           paramDict['z_max'] ,
                                           paramDict['REDSHIFT_DISTR'])


    h = hm.AngularCF(hod_model=paramDict['HOD_MODEL'], z=paramDict['z_mean'])
    h.update(hod_params          =    {"central":True})
    h.update(hmf_model           =    paramDict['HALO_MASS_FUNCTION'])
    h.update(bias_model          =    paramDict['HALO_BIAS_FUNCTION'])
    h.update(concentration_model =    paramDict['CONCENTRAION_TO_MASS_RELATION'])
    h.update(zmin                =    paramDict['z_min'])
    h.update(zmax                =    paramDict['z_max'])
    h.update(znum                =    paramDict['z_num'])
    h.update(p1                  =    redshift_distribution)
    h.update(theta_min           =    paramDict['theta_min']*(np.pi/180.0))
    h.update(theta_max           =    paramDict['theta_max']*(np.pi/180.0))
    h.update(theta_num           =    paramDict['theta_num'])
    h.update(logu_min            =    paramDict['logu_min'])
    h.update(logu_max            =    paramDict['logu_max'])
    h.update(unum                =    paramDict['unum'])
    h.update(rmin                =    paramDict['rmin'])
    h.update(rmax                =    paramDict['rmax'])
    h.update(rnum                =    paramDict['rnum'])


    if h.rmin > h.r.min():
        rmin_modified = h.r.min()
        h.update(rmin = rmin_modified)
    if h.rmax < h.r.max():
        rmax_modified = h.r.max()
        h.update(rmax = rmax_modified+5.)


    cosmo_model = default_cosmology.get_cosmology_from_string(paramDict['COSMOLOGY'])
    h.update(cosmo_model         =    cosmo_model)
    stdout('[MODEL] halo model is set ... Assume %s model' % paramDict['HOD_MODEL'])
    
    if paramDict['LITTLE_H_INCUSION']:
        little_h = h.cosmo.H0.value/100.
    else:
        little_h = 1.
    stdout('[UNIT] Include little_h or not : %s' % paramDict['LITTLE_H_INCUSION'])

    if paramDict['HOD_MODEL'] == 'Zheng05':
        h.update(hod_params={'M_min'    : paramDict['log_Mmin'] + np.log10(little_h) ,
                             'M_1'      : paramDict['log_Msat'] + np.log10(little_h) ,
                             'alpha'    : paramDict['alpha'   ]                      ,
                             'sig_logm' : paramDict['sigma'   ]                      ,
                             'M_0'      : paramDict['log_Mcut'] + np.log10(little_h)
                             })
    if paramDict['HOD_MODEL'] == 'Contreras13':
        h.update(hod_params={'M_min'    : paramDict['log_Mc'  ] + np.log10(little_h) ,
                             'M_1'      : paramDict['log_Mmin'] + np.log10(little_h) ,
                             'alpha'    : paramDict['alpha'   ]                      ,
                             'sig_logm' : paramDict['sigma'   ]                      ,
                             'fca'      : paramDict['Fca'     ]                      ,
                             'fcb'      : paramDict['Fcb'     ]                      ,
                             'fs'       : paramDict['Fs'      ]                      ,
                             'delta'    : paramDict['delta'   ]                      ,
                             'x'        : paramDict['x'       ]
                             })
    
    # get results from halomod
    modelm       = h.m/little_h                 # mass variable
    modelNcen    = h.n_cen                      # central part of occupation number
    modelNsat    = h.n_sat                      # satellite part of occupation number
    modelNtot    = h.n_tot                      # total occupation number (central + satellite)
    modelSep     = np.degrees(h.theta)          # theta variable (separation)
    modelACF     = h.angular_corr_gal           # angular correlation function
    modelACF_spl = spline(modelSep, modelACF)   # a callable function of acf

    modelSepTerm_spl_list = []
    for corr in [h.corr_gg_1h_cs, h.corr_gg_1h_ss, h.corr_gg_1h, h.corr_gg_2h]:
        corr_spl = spline(h.r, corr) # a callable function of 1-halo term of '3-d' correlation function
        angular  = angular_corr_gal(h.theta, corr_spl       ,
                                    redshift_distribution   ,
                                    paramDict['z_min']      ,
                                    paramDict['z_max']      ,
                                    paramDict['logu_min']   ,
                                    paramDict['logu_max']   ,
                                    paramDict['z_num']      ,
                                    paramDict['unum']       ,
                                    cosmo=cosmo_model
                                    ) # angular correlation function of 1-halo '3-d' correlation function
        modelSepTerm_spl_list.append(spline(np.degrees(h.theta), angular))


    if not paramDict['MODEL_ONLY']:
        outSep = obsSep
        if paramDict['APPLY_INTEGRAL_CONSTRAIN']:
            ic = np.sum(obsRR * modelACF_spl(obsSep)) / np.sum(obsRR)
        else:
            ic = 0.
    else:
        outSep = modelSep
        if paramDict['APPLY_INTEGRAL_CONSTRAIN']:
            stdout("[ERROR] Need the input acf file (including RR counts) for the calculation of I.C.")
            exit()
        else:
            ic = 0.

    outACF    = modelACF_spl(outSep) - ic
    outACF_cs = modelSepTerm_spl_list[0](outSep) - ic
    outACF_ss = modelSepTerm_spl_list[1](outSep) - ic
    outACF_1h = modelSepTerm_spl_list[2](outSep) - ic
    outACF_2h = modelSepTerm_spl_list[3](outSep) - ic

    modelNdens = h.mean_gal_den*(little_h**3)
    effBias    = h.bias_effective
    fsat       = h.satellite_fraction
    effMass    = h.mass_effective-np.log10(little_h)

    stdout('[MODEL] number density     = %f <h^3/Mpc^3>' % modelNdens)
    stdout('[MODEL] effective mass     = %f <Msun/h> in log scales' % effMass)
    stdout('[MODEL] effective bias     = %f' % effBias)
    stdout('[MODEL] satellite fraction = %f' % fsat)

    np.savetxt(paramDict['OUTPUT_ACF_PTS']+'.'+version+'.dat',
               np.c_[outSep, outACF, outACF_cs, outACF_ss, outACF_1h, outACF_2h],
               header = '--- Derived parameters ---\n'                               +\
                        'number density     = %f <h^3/Mpc^3>\n' % modelNdens         +\
                        'effective mass     = %f <Msun/h> in log scales\n' % effMass +\
                        'effective bias     = %f\n' % effBias                        +\
                        'satellite fraction = %f\n' % fsat                           +\
                        'IC                 = %.4f\n' % ic                           +\
                        '\n'                                                         +\
                        'sep(deg) omega 1h_cs 1h_ss 1h 2h'
    )
    np.savetxt(paramDict['OUTPUT_HON_PTS']+'.'+version+'.dat', np.c_[np.log10(h.m), modelNtot, modelNcen, modelNsat],
        header='# log(M) N_tot N_cen N_sat'
    )

    if not paramDict['PTS_ONLY']:
        stdout('[PLOTS] Additionally, output an ACF plot and a HON plot')

        # plot acf w/o obs. data
        plt.figure()
        plt.plot(outSep, outACF,    '-',  label='overall (w/ I.C.)')
        plt.plot(outSep, outACF_1h, '--', label='1-halo (w/ I.C.)')
        plt.plot(outSep, outACF_2h, '--', label='2-halo (w/ I.C.)')
        if not paramDict['MODEL_ONLY']:
            plt.errorbar(obsSep, obsACF, yerr=obsACFerr, fmt='o', label='data')
        plt.xlabel('$\\theta\ [deg]$',  fontsize=20)
        plt.ylabel('$\omega(\\theta)$', fontsize=20)
        plt.legend(loc='best',          fontsize=15)
        plt.loglog()
        plt.savefig(paramDict['OUTPUT_ACF_PLOT']+'.'+version+'.png')

        # plot hon
        plt.figure()
        plt.plot(modelm, modelNtot, label='total')
        plt.plot(modelm, modelNcen, label='Central')
        plt.plot(modelm, modelNsat, label='Satellite')
        plt.xlabel('$M_h [M_\dot]$',  fontsize=20)
        plt.ylabel('$HON$',           fontsize=20)
        plt.legend(loc='best',        fontsize=15)
        plt.loglog()
        plt.savefig(paramDict['OUTPUT_HON_PLOT']+'.'+version+'.png')

    stdout('[END]')
