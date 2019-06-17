# This script uses variables without little-h, but halomod uses units with little-h (Mpc versus Mpc/h etc.), so there are conversions throughout

## ==STANDARD MODULES==
import random
import time, sys, re
import numpy as np
from scipy.interpolate import\
    InterpolatedUnivariateSpline as spline
from astropy.cosmology import default_cosmology
from collections import OrderedDict

## ==HALO MODEL & FITTING MODULES==
import halomod as hm
import emcee
from emcee.interruptible_pool import InterruptiblePool

# sections in main() function
# (1) [_MON]: load in tables
# (2) [_TUE]: initialize halo model using halomod
# (3) [_WED]: build probability functions for running mcmc
# (4) [_THU]: generate inital points from random
# (5) [_FRI]: MAIN PART for mcmc process
# (6) [_SAT]: based on mcmc samples, derive some parameters like effective bias
# (7) [_SUN]: save results to files

if __name__ == '__main__':

    def GetParams():
    
        params = InitializeParameters()
    
        filename = GetParamFilename()
        fileParams = np.loadtxt(filename, dtype=np.str, usecols=(0,1))
        fileParamsDict = {}
        for idx in range(fileParams.shape[0]):
            fileParamsDict[fileParams[idx,0]] = fileParams[idx,1]
    
        diffFromInit = set(params.keys()).difference(fileParamsDict.keys())
        diffFromInit = list(diffFromInit)

        # this list contains hod parameters specifying the range of input parameters,
        # these parameters are flexible enough to accept numerical variables or alphabetic ones.
        if fileParamsDict['HOD_MODEL'] == 'Zheng05':
            flexParams, _ = FlexParameters()
            flexParamsName = flexParams.keys()
        if fileParamsDict['HOD_MODEL'] == 'Contreras13':
            _, flexParams = FlexParameters()
            flexParamsName = flexParams.keys()
    
        for pname, pvalue in params.items():
            if pname in diffFromInit:
                # if a parameter is not found in the file,
                # this code will take the default value set in
                # InitializeParameters() function
                pass
            else:
                if pname in flexParamsName: # check those flexible parameters
                    if ParamIsAlphabetic(fileParamsDict[pname]):
                        params[pname] = fileParamsDict[pname]
                    else:
                        params[pname] = fileParamsDict[pname].astype(type(pvalue))
                else:
                    params[pname] = fileParamsDict[pname].astype(type(pvalue))
    
        return params
    
    def GetParamFilename():
    
        if len(sys.argv) == 2:
            stdout('[FILE] read parameter file: %s ' % sys.argv[1])
            return sys.argv[1]
        else:
            sys.exit(
                '%s... please input a parameter file by using command : \
                $python halo_modeling [YOUR_PARAMETER_FILES]' % sys.argv[0]
            )

    def _checkFlexParams(paramDict):
        mainDict, flexDict = paramDict

        ZhengParams = ['log_Mmin', 'log_Msat', 'alpha', 'sigma', 'log_Mcut']
        ContrParams = ['log_Mc', 'log_Mmin', 'alpha', 'sigma', 'Fca', 'Fcb',
                       'Fs', 'delta', 'x']

        for key, valueList in flexDict:
            if mainDict[key] != key:
                break
        return
        
    def CheckConstantFlex(paramDict):
        mainDict, flexDict = paramDict

        premade = {}

        for key, valuelist in flexDict.items():
            if np.abs(mainDict[valuelist[0]] - mainDict[valuelist[1]]) < 1e-5:
                premade[key] = mainDict[valuelist[0]]
            else:
                premade[key] = -99.

        return premade


    def FlexParameters():

        return OrderedDict([ ('log_Mmin' , ['log_Mmin_min' , 'log_Mmin_max']),
                             ('log_Msat' , ['log_Msat_min' , 'log_Msat_max']),
                             ('alpha'    , ['alpha_min'    , 'alpha_max'   ]),
                             ('sigma'    , ['sigma_min'    , 'sigma_max'   ]),
                             ('log_Mcut' , ['log_Mcut_min' , 'log_Mcut_max'])]),\
               OrderedDict([ ('log_Mc'   , ['log_Mc_min'   , 'log_Mc_max'  ]) ,
                             ('log_Mmin' , ['log_Mmin_min' , 'log_Mmin_max']) ,
                             ('alpha'    , ['alpha_min'    , 'alpha_max'   ]) ,
                             ('sigma'    , ['sigma_min'    , 'sigma_max'   ]) ,
                             ('Fca'      , ['Fca_min'      , 'Fca_max'     ]) ,
                             ('Fcb'      , ['Fcb_min'      , 'Fcb_max'     ]) ,
                             ('Fs'       , ['Fs_min'       , 'Fs_max'      ]) ,
                             ('delta'    , ['delta_min'    , 'delta_max'   ]) ,
                             ('x'        , ['x_min'        , 'x_max'       ])])
    
    def InitializeParameters():
    
        return {
                # FILES
                'WORKING_DIRECTORY':      'YOUR_WORKING_DIRECTORY',
                'INPUT_ACF':              'YOUR_CORRELATION_FILE',
                'INPUT_ZD':               'YOUR_REDSHIFT_DISTRIBUTION',
                'INPUT_COV':              'YOUR_COVARIANCE_MATRIX',
                'OUTPUT_MCMC_SAMPLES':    'MCMC_CHAIN_SAMPLE',
                'OUTPUT_ACF_SAMPLES':     'BEST_FIT_ACF_SAMPLE',
                'OUTPUT_DERIVED_SAMPLES': 'DERIVED_PARAMETER_SAMPLE',
                # MODELS
                'VERSION':                       'v1',
                'COSMOLOGY':                     'Planck15',
                'HOD_MODEL':                     'Zheng05',
                'HALO_MASS_FUNCTION':            'Tinker10',
                'HALO_BIAS_FUNCTION':            'Tinker10',
                'CONCENTRAION_TO_MASS_RELATION': 'Duffy08',
                # SWITCHES
                'APPLY_INTEGRAL_CONSTRAIN': True,
                'LITTLE_H_INCUSION':        True,
                # HOD settings
                'obs_number_density': 0.0005,
                'err_obs_ndens':      0.00002,
                'z_mean':             1.12,
                'z_min':              1.0,
                'z_max':              1.25,
                'z_num':              100,
                'logM_min':           6,
                'logM_max':           16,
                'theta_min':          1./3600.,
                'theta_max':          3600./3600.,
                'theta_num':          60,
                'logu_min':           -5.,
                'logu_max':           2.5,
                'unum':               150,
                # parameters
                'log_Mmin_min':       11.,
                'log_Mmin_max':       13.,
                'log_Msat_min':       13.,
                'log_Msat_max':       14.,
                'log_Mcut_min':       9.,
                'log_Mcut_max':       10.,
                'alpha_min':          0.7,
                'alpha_max':          1.35,
                'sigma_min':          0.25,
                'sigma_max':          0.6,
                'log_Mc_min':         10.,
                'log_Mc_max':         13.,
                'Fca_min'   :         0.02,
                'Fca_max'   :         1.5,
                'Fcb_min'   :         0.02,
                'Fcb_max'   :         1.5,
                'Fs_min'    :         0.8,
                'Fs_max'    :         1.2,
                'delta_min' :         0.9,
                'delta_max' :         1.05,
                'x_min'     :         0.9,
                'x_max'     :         1.05,
                # parameter flags
                'log_Mmin' : 'log_Mmin' ,
                'log_Msat' : 'log_Msat' ,
                'log_Mcut' : 'log_Mcut' ,
                'alpha'    : 'alpha'    ,
                'sigma'    : 'sigma'    ,
                'log_Mc'   : 'log_Mc'   ,
                'Fca'      : 'Fca'      ,
                'Fcb'      : 'Fcb'      ,
                'Fs'       : 'Fs'       ,
                'delta'    : 'delta'    ,
                'x'        : 'x'        ,
                # MCMC settings
                'mcmc_steps':         1500,
                'Ndim':               5,
                'Nwalkers':           20,
                'sample_rate':        10,
                'burnin_rate':        0.25,
                'Nprocessors':        1
               }

    def ParamIsAlphabetic(string):
        # this function will check if the 1st characters of the string is
        # alphabetic

        return bool(re.search('^[a-zA-z]', string))

    def stdout(message):

        print('%s...  %s' % (sys.argv[0], message))

    # load in parameters from .param file
    paramDict = GetParams()
    
    # initialize flex parameter dictionary
    if paramDict['HOD_MODEL'] == 'Zheng05':
        flexParamDict, _ = FlexParameters()
    if paramDict['HOD_MODEL'] == 'Contreras13':
        _, flexParamDict = FlexParameters()

    hodParDictPremade = CheckConstantFlex([paramDict, flexParamDict])
    if sum(np.array(hodParDictPremade.values())<0) != paramDict['Ndim']:
        stdout("[ERROR] Dimension of fitting parameters are not equal to"+\
               " 'Ndim' parameter")
        sys.exit("TERMINATED")

    # declare version
    version = paramDict['VERSION']
    stdout("Version : %s" % version)

    # initialize filenames             
    wd               = paramDict['WORKING_DIRECTORY']
    corrFilename     = paramDict['INPUT_ACF']
    rdFilename       = paramDict['INPUT_ZD']
    covFilename      = paramDict['INPUT_COV']
    mcmcFilename     = paramDict['OUTPUT_MCMC_SAMPLES']
    acfModelFilename = paramDict['OUTPUT_ACF_SAMPLES']
    paramsFilename   = paramDict['OUTPUT_DERIVED_SAMPLES']

    ## ==load in tables (1) [_MON]==
    # main file: contains theta, acf, RR
    data = np.genfromtxt(wd+corrFilename)
    obsSep = data[:,0]
    obsACF = data[:,1]
    obsRR  = data[:,2]
    
    nz = np.loadtxt(wd+rdFilename)
    redshift_distribution = spline(nz[:,0], nz[:,1])

    cov = np.loadtxt(wd+covFilename)
    invCov = np.linalg.inv(cov)

    stdout('Files are successfully loaded')
    ## ==finish loading==
    
    ## ==initialize halo model (2) [_TUE]==
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

    cosmo_model = default_cosmology.get_cosmology_from_string(paramDict['COSMOLOGY'])
    h.update(cosmo_model         =    cosmo_model)
    
    if paramDict['LITTLE_H_INCUSION']:
        little_h = h.cosmo.H0.value/100.
    else:
        little_h = 1.
    stdout('Halo model is established')
    ## ==halo model established==

    ## ==build probability functions for mcmc (3) [_WED]==

    # Likelihood function in log scale
    #def LnLikeli(th, obsSep, obsACF, obsRR, invCov, paramDict, premade):
    def LnLikeli(th, obsSep, obsACF, obsRR, invCov, paramDict):
        #for ik, key in enumerate(premade.keys()):
        #    if premade[key] < 0:
        #        premade[key] = th[ik]

        #h.update(hod_params = premade)

        if paramDict['HOD_MODEL'] == 'Zheng05':
            M_min, M_1, alpha, sig_logm, M_0 = th
            h.update(hod_params={"M_min":    M_min + np.log10(little_h),
                                 "M_1":      M_1   + np.log10(little_h),
                                 "alpha":    alpha,
                                 "sig_logm": sig_logm,
                                 "M_0":      M_0   + np.log10(little_h)})

        if paramDict['HOD_MODEL'] == 'Contreras13':
            M_c, M_min, alpha, sig_logm, Fca, Fcb, Fs, delta, x = th
            h.update(hod_params={'M_min'    : M_c   + np.log10(little_h) ,
                                 'M_1'      : M_min + np.log10(little_h) ,
                                 'alpha'    : alpha                     ,
                                 'sig_logm' : sig_logm                  ,
                                 'fca'      : Fca                       ,
                                 'fcb'      : Fcb                       ,
                                 'fs'       : Fs                        ,
                                 'delta'    : delta                     ,
                                 'x'        : x       
                                 })
        
        modelSep     = np.degrees(h.theta)
        modelACF     = h.angular_corr_gal
        modelACF_spl = spline(modelSep, modelACF, k=3) # callable function
        
        if paramDict['APPLY_INTEGRAL_CONSTRAIN']:
            ic = np.sum(obsRR * modelACF_spl(obsSep)) / np.sum(obsRR)
        else:
            ic = 0
            
        modelACF   = modelACF_spl(obsSep) - ic
        modelNdens = h.mean_gal_den*(little_h**3)

        diffACF = obsACF - modelACF
        lnACFLike   = -0.5 * np.sum(diffACF[:,np.newaxis] * invCov* diffACF)
        lnNdensLike = -0.5 * ((modelNdens - paramDict['obs_number_density'])/paramDict['err_obs_ndens']) *\
                             ((modelNdens - paramDict['obs_number_density'])/paramDict['err_obs_ndens'])
    
        #obsACF_model_difference_trans = np.transpose(obsACF_model_difference)
        #log_liklihood_clustering = -0.5 * (obsACF_model.dot(cov_matrix_inverse.dot(obsACF_model_trans)))
        #log_L_acf = -0.5 * np.sum(np.power((obsACF - model)/obsACFerr, 2.))
        #log_L_n   = -0.5 * np.power((model_num_density - obsNdens)/obsNdensErr, 2.)
        return lnACFLike + lnNdensLike
        
    # Prior probability function in log scale
    def LnPrior(th, paramDict):
        #mainDict, flexDict = paramDict

        #if mainDict['HOD_MODEL'] == 'Zheng05':
        if paramDict['HOD_MODEL'] == 'Zheng05':
            #M_min, M_1, alpha, sig_logm, M_0 = th
            
            if  paramDict['log_Mmin_min'] < th[0] < paramDict['log_Mmin_max'] and \
                M_min                     < th[1] < paramDict['log_Msat_max'] and \
                paramDict['alpha_min'   ] < th[2] < paramDict['alpha_max'   ] and \
                paramDict['sigma_min'   ] < th[3] < paramDict['sigma_max'   ] and \
                paramDict['log_Mcut_min'] < th[4] < M_1 : 
                return 0.0
            else:
                return -np.inf

        #if mainDict['HOD_MODEL'] == 'Contreras13':
        if paramDict['HOD_MODEL'] == 'Contreras13':
            M_c, M_min, alpha, sig_logm, Fca, Fcb, Fs, delta, x = th
            
            if  paramDict['log_Mc_min'  ] < M_c      < paramDict['log_Mc_max'  ] and \
                paramDict['log_Mmin_min'] < M_min    < paramDict['log_Mmin_max'] and \
                paramDict['alpha_min'   ] < alpha    < paramDict['alpha_max'   ] and \
                paramDict['sigma_min'   ] < sig_logm < paramDict['sigma_max'   ] and \
                paramDict['Fca_min'     ] < Fca      < paramDict['Fca_max'     ] and \
                paramDict['Fcb_min'     ] < Fcb      < paramDict['Fcb_max'     ] and \
                paramDict['Fs_min'      ] < Fs       < paramDict['Fs_max'      ] and \
                paramDict['delta_min'   ] < delta    < paramDict['delta_max'   ] and \
                paramDict['x_min'       ] < x        < paramDict['x_max'       ] :
                return 0.0
            else:
                return -np.inf

    # Log posterior
    def LnProb(th, obsSep, obsACF, obsRR, invCov, paramDict):

        if paramDict['HOD_MODEL'] == 'Zheng05':
            prior = LnPriorZheng(th, paramDict)

        if paramDict['HOD_MODEL'] == 'Contreras13':
            prior = LnPriorContreras(th, paramDict)

        if not np.isfinite(prior):
            return -np.inf
        return prior + LnLikeli(th, obsSep, obsACF, obsRR, invCov, paramDict)

    stdout('Probability functions are defined')
    ## ==functions defined==
    
    ## ==generate initial points for mcmc (4) [_THU]==
    mcmcNumSteps = paramDict['mcmc_steps']
    ndim         = paramDict['Ndim']
    nwalkers     = paramDict['Nwalkers']
    sampleRate   = paramDict['sample_rate']
    burninRate   = paramDict['burnin_rate']
    nprocessors  = paramDict['Nprocessors']

    ipoints = []

    if paramDict['HOD_MODEL'] == 'Zheng05':
        for i in range(nwalkers):
            rand1 = random.uniform(paramDict['log_Mmin_min'], paramDict['log_Mmin_max'])
            rand2 = random.uniform(rand1                    , paramDict['log_Msat_max'])
            rand3 = random.uniform(paramDict['alpha_min']   , paramDict['alpha_max'])
            rand4 = random.uniform(paramDict['sigma_min']   , paramDict['sigma_max'])
            rand5 = random.uniform(paramDict['log_Mcut_min'], rand2)
            ipoints.append([rand1,rand2,rand3,rand4,rand5])
    
    if paramDict['HOD_MODEL'] == 'Contreras13':
        for i in range(nwalkers):
            rand1 = random.uniform(paramDict['log_Mc_min'  ], paramDict['log_Mc_max'  ])
            rand2 = random.uniform(paramDict['log_Mmin_min'], paramDict['log_Mmin_max'])
            rand3 = random.uniform(paramDict['alpha_min'   ], paramDict['alpha_max'   ])
            rand4 = random.uniform(paramDict['sigma_min'   ], paramDict['sigma_max'   ])
            rand5 = random.uniform(paramDict['Fca_min'     ], paramDict['Fca_max'     ])
            rand6 = random.uniform(paramDict['Fcb_min'     ], paramDict['Fcb_max'     ])
            rand7 = random.uniform(paramDict['Fs_min'      ], paramDict['Fs_max'      ])
            rand8 = random.uniform(paramDict['delta_min'   ], paramDict['delta_max'   ])
            rand9 = random.uniform(paramDict['x_min'       ], paramDict['x_max'       ])
            ipoints.append([rand1,rand2,rand3,rand4,rand5,rand6,rand7,rand8,rand9])

    stdout('Initialized mcmc points')
    ## ==done==
    
    ## ==main part for running mcmc (5) [_FRI]==
    pool = InterruptiblePool(processes=nprocessors)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        LnProb,
        args=(obsSep, obsACF, obsRR, invCov, paramDict),
        pool=pool
    )
    stdout('MCMC sampler is created')
    
    t1 = time.time()
    sampler.run_mcmc(ipoints, mcmcNumSteps) 
    t2 = time.time()
    stdout('MCMC Finished')
    stdout('MCMC took '+str(np.floor((t2-t1)/60))+' minutes')
    pool.close()
    
    nBurnin = int(mcmcNumSteps*burninRate)

    samples = sampler.chain[:,nBurnin:,:].reshape((-1, ndim))
    LnProb  = sampler.lnprobability[:,nBurnin:].reshape(-1)
    ## ==done mcmc==
    
    ## ==now start to derive some parameters (6) [_SAT]==
    #nsamples = nwalkers*mcmcNumSteps/sampleRate
    nsamples = samples.shape[0]/sampleRate
    modelACFDistr       = np.zeros(shape=(len(obsSep), nsamples))
    modelACFCompDistr   = np.zeros(shape=(4, len(obsSep), nsamples))
    effBiasDistr        = np.zeros(nsamples)
    effMassDistr        = np.zeros(nsamples)
    fsatDistr           = np.zeros(nsamples)
    nDensModelDistr     = np.zeros(nsamples)
    
    for i in range(nsamples):
        
        if paramDict['HOD_MODEL'] == 'Zheng05':
            h.update(hod_params={"M_min"    : samples[i*sampleRate, 0] + np.log10(little_h) ,
                                 "M_1"      : samples[i*sampleRate, 1] + np.log10(little_h) ,
                                 "alpha"    : samples[i*sampleRate, 2]                      ,
                                 "sig_logm" : samples[i*sampleRate, 3]                      ,
                                 "M_0"      : samples[i*sampleRate, 4] + np.log10(little_h)
                                 })

        if paramDict['HOD_MODEL'] == 'Contreras13':
            h.update(hod_params={'M_min'    : samples[i*sampleRate, 0] + np.log10(little_h) ,
                                 'M_1'      : samples[i*sampleRate, 1] + np.log10(little_h) ,
                                 'alpha'    : samples[i*sampleRate, 2]                      ,
                                 'sig_logm' : samples[i*sampleRate, 3]                      ,
                                 'fca'      : samples[i*sampleRate, 4]                      ,
                                 'fcb'      : samples[i*sampleRate, 5]                      ,
                                 'fs'       : samples[i*sampleRate, 6]                      ,
                                 'delta'    : samples[i*sampleRate, 7]                      ,
                                 'x'        : samples[i*sampleRate, 8]
                                 })

        
        modelSep     = np.degrees(h.theta)
        modelACF     = h.angular_corr_gal
        modelACF_spl = spline(modelSep, modelACF, k=3)
        
        if paramDict['APPLY_INTEGRAL_CONSTRAIN']:
            ic = np.sum(obsRR * modelACF_spl(obsSep)) / np.sum(obsRR)
        else:
            ic = 0

        modelACF = modelACF_spl(obsSep) - ic
        
        modelACFDistr[:,i] = modelACF
        effBiasDistr[i]    = h.bias_effective
        fsatDistr[i]       = h.satellite_fraction
        effMassDistr[i]    = h.mass_effective-np.log10(little_h)
        nDensModelDistr[i] = h.mean_gal_den*(little_h**3)

        for ic, corr in enumerate[h.corr_gg_1h_cs, h.corr_gg_1h_ss, h.corr_gg_1h, h.corr_gg_2h]:
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
            ang_spl = spline(np.degrees(h.theta), angular)
            modelACFCompDistr[ic,:,i] = ang_spl(outSep) - ic

    
    # Create objects for holding the models
    modelACF_best  = np.zeros(len(obsSep))
    modelACF_lower = np.zeros(len(obsSep))
    modelACF_upper = np.zeros(len(obsSep))

    modelACFComp_best  = np.zeros((4, len(obsSep)))
    modelACFComp_lower = np.zeros((4, len(obsSep)))
    modelACFComp_upper = np.zeros((4, len(obsSep)))
    
    # Find percentiles of acf
    for i in range(len(obsSep)):
        modelACF_best[i]  = np.percentile(modelACFDistr[i,:],50)
        modelACF_lower[i] = np.percentile(modelACFDistr[i,:],16)
        modelACF_upper[i] = np.percentile(modelACFDistr[i,:],84)

        for j in range(4):
            modelACFComp_best[j, i]  = np.percentile(modelACFCompDistr[j,i,:], 50)
            modelACFComp_lower[j, i] = np.percentile(modelACFCompDistr[j,i,:], 50)
            modelACFComp_upper[j, i] = np.percentile(modelACFCompDistr[j,i,:], 50)
    ## ==parameters are derived==
    
    
    ## ==save all derived parameters to file (7) [_SUN]==
    
    np.savetxt(wd+mcmcFilename+version+".dat", samples) # Save HOD param samples
    
    derrived_parameters = np.transpose([fsatDistr, effBiasDistr, effMassDistr, nDensModelDistr])
    np.savetxt(wd+paramsFilename+version+".dat", derrived_parameters)
    
    model_acf = np.transpose([obsSep,
                              modelACF_lower       , modelACF_best       , modelACF_upper       ,
                              modelACFComp_lower[0], modelACFComp_best[0], modelACFComp_upper[0],
                              modelACFComp_lower[1], modelACFComp_best[1], modelACFComp_upper[1],
                              modelACFComp_lower[2], modelACFComp_best[2], modelACFComp_upper[2],
                              modelACFComp_lower[3], modelACFComp_best[3], modelACFComp_upper[3]])
    np.savetxt(wd+acfModelFilename+version+".dat",
               model_acf,
               header='sep tot_low tot_best tot_up'+\
                      ' cs_low cs_best cs_up'      +\
                      ' ss_low ss_best ss_up'      +\
                      ' 1h_low 1h_best 1h_up'      +\
                      ' 2h_low 2h_best 2h_up') # Save model acfs

    ##################################
    ##### end of main() function #####
    ##################################
