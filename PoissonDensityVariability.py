import sys, os
import re

# This script uses the Python library DFNLab for generation and analysis of DFNs (https://github.com/FractoryLabcom/software  )
import dfnlab.GeometryLib as dfn_geo
import dfnlab.DFNBasis as dfn
import dfnlab.DFNGenerator as dfn_gen
import dfnlab.DFNAnalysis as dfn_analysis

import time
import subprocess
import multiprocessing as multi_process
from multiprocessing import Process, Manager, Pool
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as pyplot 
import matplotlib.legend_handler
import matplotlib.ticker as mticker
import mplstereonet as mpl_stereo
import pylab as pl
import math, numpy as np
import sympy
from sympy import lambdify
import argparse
import numpy

##
# We generate here Poissonian models with differents size distribution exponent to study the impact of this exponent on lacunarity
##

####################### IO tools ################################

def name_size_folder(root,L):
    simulationPath = root + '\simulations'
    if not os.path.exists(simulationPath):
        os.makedirs(simulationPath) 
    os.chdir(simulationPath)
    LPath = simulationPath + '\L=' + str(L)
    if not os.path.exists(LPath):
        os.makedirs(LPath)
    return LPath

def name_p32_folder(root,L,p32):
    LPath = name_size_folder(root,L)
    p32Path = LPath + '\p32=' + str(p32)
    if not os.path.exists(p32Path):
        os.makedirs(p32Path)
    os.chdir(p32Path)
    return p32Path

def name_orientation_folder(root,L,p32,orientation):
    p32Path = name_p32_folder(root,L,p32)
    dip = orientation[0]
    dipdir = orientation[1]
    kappa = orientation[2]
    orientationPath = p32Path + '\dip=' + str(dip) + '_dipdir=' + str(dipdir) + '_kappa=' + str(kappa)
    if not os.path.exists(orientationPath):
        os.makedirs(orientationPath)
    os.chdir(orientationPath)
    return orientationPath


def name_simulation_folder(root,L,p32,orientation,a,lmin,lmax):
    orientationPath = name_orientation_folder(root,L,p32,orientation)
    exponentPath = orientationPath + r'\a=' + str(a) + r'_lmin=' + str(lmin) + r'_lmax=' + str(lmax) + r'_L=' + str(L)
    if not os.path.exists(exponentPath):
        os.makedirs(exponentPath)
    os.chdir(exponentPath)
    return exponentPath


def name_disk(root,L,p32,orientation,a,s,lmin,lmax):
    simu_path = name_simulation_folder(root,L,p32,orientation,a,lmin,lmax)
    dip = orientation[0]
    dipdir = orientation[1]
    kappa = orientation[2]
    name_disk = simu_path +  r'\poisson_L=' + str(L) + '_p32=' + str(p32) + '_dip=' + str(dip) + '_dipdir=' + str(dipdir) + '_kappa=' + str(kappa) + '_a=' + str(a)  + '_lmin=' + str(lmin) + '_lmax=' + str(lmax) + '_seed=' + str(s)+ '.disk'
    return name_disk


def chunks(theList, nbThreads):
    for i in range(0, len(theList), nbThreads):
        yield theList[i:i + nbThreads]

####################### Generation tools ################################

def generate_poisson_network(root,L,p32,orientation,a,lmin,lmax,seed):
    system = dfn.System()
    system.buildParallelepiped([0, 0, 0],(L+lmax))
    fnet = dfn.DFN(system)
    generator = dfn_gen.PoissonGenerator(fnet, seed)
    dip = orientation[0]
    dipdir = orientation[1]
    kappa = orientation[2]
    if ~np.isnan(dip) and ~np.isnan(dipdir) and ~np.isnan(kappa):
        generator.setOrientationsFisher(dip,dipdir,kappa)
    else:
        generator.setOrientationsUniform()
    generator.setPositionsUniform()
    generator.setStopDensity(p32)
    if a==0:
         generator.setSizesUniform(lmin,lmax)
    else:
        generator.setSizesPowerlaw(a,lmin,lmax)
    generator.generate()
    system2 = dfn.System()
    system2.buildParallelepiped([0, 0, 0],L)
    fnet.changeSystem(system2)
    nameDisk = name_disk(root,L,p32,orientation,a,seed,lmin,lmax)
    print(nameDisk)
    dfn.write_disk_file(fnet,nameDisk)


def generate_networks(root,sList,systemSizeList,p32List,orientationList,exponentList,lmin,lmax,nbThreads):
    processes=[]
    # Generate...
    # ... for each system size...
    for L in systemSizeList:
        # ... with different orientations
        for orientation in orientationList:
            # ... with different densities
            for p32 in p32List:
                # ... several realizations
                for seed in sList:
                    # ... with different exponents
                    for a in exponentList:
                        p = multi_process.Process(target=generate_poisson_network, args=(root,L,p32,orientation,a,lmin,lmax,seed))
                        processes.append(p)
                    # ...and constant sizes with l<<L
                    lSmall = 0.05*L
                    p = multi_process.Process(target=generate_poisson_network, args=(root,L,p32,orientation,0,lSmall,lSmall,seed))
                    processes.append(p)
                    # ... constant size networks with l>>L
                    lBig = L
                    p = multi_process.Process(target=generate_poisson_network, args=(root,L,p32,orientation,0,lBig,lBig,seed))
                    processes.append(p)

    for i in chunks(processes,nbThreads):
        for j in i:
            j.start()
        for j in i:
            j.join()



####################### Analysis tools ############################

def launch_size_distribution_analysis_on_networks(root,systemSizeList,p32List,orientationList,exponentList,sList,lmin,lmax,nbThreads,nbin):
    processes = []
    for L in systemSizeList:
        for p32 in p32List:
            for orientation in orientationList:
                for a in exponentList:
                    folder = name_simulation_folder(root,L,p32,orientation,a,lmin,lmax)
                    for seed in sList:
                        simu_folder = name_simulation_folder(root,L,p32,orientation,a,lmin,lmax)
                        full_disk_file = name_disk(root,L,p32,orientation,a,seed,lmin,lmax)
                        disk_file = full_disk_file[len(simu_folder)+1:]
                        p = multi_process.Process(target=launch_size_distribution_analysis, args=(folder,disk_file,L,nbin))
                        processes.append(p)

    for i in chunks(processes,nbThreads):
        for j in i:
            j.start()
        for j in i:
            j.join()



def launch_size_distribution_analysis(folder,disk_file,L,nbin):
    os.chdir(folder)
    nl_path = folder + r'\nl'
    if not os.path.exists(nl_path):
        os.makedirs(nl_path)
    nl_name = nl_path +  r'\nl_' + disk_file[:-5] + '.dat'
    system = dfn.System()
    system.buildParallelepiped([0, 0, 0],L) 
    fnet = dfn.DFN(system)
    dfn.load_disk_file(fnet,disk_file, True) 
    analyser = dfn_analysis.DFNAnalyser(fnet)
    nl = analyser.sizeDistribution(nbin);
    dfn_analysis.DFNAnalyser.write_distribution(nl,nl_name)


def get_size_distribution(root,L,p32,orientation,a,lmin,lmax):
    simu_path = name_simulation_folder(root,L,p32,orientation,a,lmin,lmax)
    nl_path = simu_path + r'\nl'
    os.chdir(nl_path)
    l_tot=[]
    nl_tot=[]
    for nl_filename in os.listdir(nl_path):
        if os.path.isfile(nl_filename)==False:
            continue
        nl_file = open(nl_filename,'r')
        lines = nl_file.readlines() # title
        l=[]
        nl=[]
        for i in range(1,len(lines)):
            line = lines[i]
            size = float(line.split('\t')[2])
            nb = float(line.split('\t')[1])
            l.append(size if size!=0 else np.nan)
            nl.append(nb if nb!=0 else np.nan)
        l_tot.append(l)
        nl_tot.append(nl)
    l_tot=np.asarray(l_tot)
    l_mean= np.nanmean(l_tot,axis=0)
    nl_tot=np.asarray(nl_tot)
    nl_mean= np.nanmean(nl_tot,axis=0)
    nl_std= np.nanstd(nl_tot,axis=0)
    size_distribution = [l_mean,nl_mean,nl_std]
    return size_distribution


def get_size_distribution_fit(root,L,p32,orientation,a,lmin,lmax):
    size_distribution = get_size_distribution(root,L,p32,orientation,a,lmin,lmax)
    log_l = np.log10(size_distribution[0])
    log_nl = np.log10(size_distribution[1])
    idx = np.isfinite(log_l) & np.isfinite(log_nl)
    alpha = np.mean(log_nl[idx]-(-a)*log_l[idx])
    return (a,10**alpha)



def launch_lacunarity_analysis_on_networks(root,systemSizeList,p32List,orientationList,exponentList,sList,lmin,lmax,nbThreads,smin):
    processes = []
    for L in systemSizeList:
        for p32 in p32List:
            for orientation in orientationList:
                for a in exponentList:
                    for seed in sList:
                        simu_folder = name_simulation_folder(root,L,p32,orientation,a,lmin,lmax)
                        full_disk_file = name_disk(root,L,p32,orientation,a,seed,lmin,lmax)
                        disk_file = full_disk_file[len(simu_folder)+1:]
                        p = multi_process.Process(target=launch_lacunarity_analysis, args=(simu_folder,disk_file,L,smin))
                        processes.append(p)

                lSmall = 0.05*L
                for seed in sList:
                    simu_folder = name_simulation_folder(root,L,p32,orientation,0,lSmall,lSmall)
                    full_disk_file = name_disk(root,L,p32,orientation,0,seed,lSmall,lSmall)
                    disk_file = full_disk_file[len(simu_folder)+1:]
                    p = multi_process.Process(target=launch_lacunarity_analysis, args=(simu_folder,disk_file,L,smin))
                    processes.append(p)

                lBig = L
                for seed in sList:
                    simu_folder = name_simulation_folder(root,L,p32,orientation,0,lBig,lBig)
                    full_disk_file = name_disk(root,L,p32,orientation,0,seed,lBig,lBig)
                    disk_file = full_disk_file[len(simu_folder)+1:]
                    p = multi_process.Process(target=launch_lacunarity_analysis, args=(simu_folder,disk_file,L,smin))
                    processes.append(p)

    for i in chunks(processes,nbThreads):
        for j in i:
            j.start()
        for j in i:
            j.join()


def launch_lacunarity_analysis(folder,disk_file,L,smin):
    os.chdir(folder)
    lacu3D_path = folder + r'\lacunarity3D'
    if not os.path.exists(lacu3D_path):
        os.makedirs(lacu3D_path)
    lacu1D_path = folder + r'\lacunarity1D'
    if not os.path.exists(lacu1D_path):
        os.makedirs(lacu1D_path)
    lacu3D_name = lacu3D_path +  r'\lacu3D_' + disk_file[:-5] + '.dat'
    lacu1D_name = lacu1D_path +  r'\lacu1D_' + disk_file[:-5] + '.dat'

    system = dfn.System()
    system.buildParallelepiped([0, 0, 0],L) 
    wells = {}
    idWell=0
    for i in range(0,9):
        x = -0.4*L + i*0.1*L
        for j in range(0,9):
            y = -0.4*L + j*0.1*L
            wells[idWell]=dfn.Well1D([x, y, 0.5*L], [x, y, -0.5*L],idWell)
            idWell+=1
    for well in wells.values():
        system.addWellTunnel(well)
    fnet = dfn.DFN(system)
    dfn.load_disk_file(fnet,disk_file,True)
    fnet.computeIntersections(1)
    analyser = dfn_analysis.DFNAnalyser(fnet)

    # Lacunarity 3D
    lacu3D = analyser.lacunarity3D(smin); 
    dfn_analysis.DFNAnalyser.write_distribution(lacu3D,lacu3D_name)

    # Lacunarity 1D
    lacuWells ={}
    for id,well in wells.items():
        lacuWells[id] = analyser.lacunarity1D(id,smin); 
    lacu1D={}
    lacu1D['scaleFactor']=lacuWells[0]['scale_factor']
    lacu1D['scale']=lacuWells[0]['scale']
    n = len(lacu1D['scale'])
    p10_mean = [0 for i in range(0,n)]
    p10_std_dev = [0 for i in range(0,n)]
    for lacuWell in lacuWells.values():
        p10_mean = [ (p10_mean[i] + lacuWell['p10_mean'][i]) for i in range(0,n)]
        p10_std_dev = [ (p10_std_dev[i] + lacuWell['p10_std_dev'][i]**2) for i in range(0,n)]
    p10_mean = [(1./float(len(wells)))*elt for elt in p10_mean]
    p10_std_dev = [math.sqrt((1./float(len(wells)))*elt) for elt in p10_std_dev]
    lacu1D['p10_mean'] = p10_mean
    lacu1D['p10_std_dev'] = p10_std_dev
    lacu1D['p10_std_dev_norm'] = [ (lacu1D['p10_std_dev'][i]/lacu1D['p10_mean'][i])  for i in range(0,n)]
    lacu1D['p10_lacunarity'] = [ lacu1D['p10_std_dev_norm'][i]**2 for i in range(0,n)]
    dfn_analysis.DFNAnalyser.write_distribution(lacu1D,lacu1D_name)


    
def get_pXX_mean(root,pXX,L,p32,orientation,a,lmin,lmax):
    simu_path = name_simulation_folder(root,L,p32,orientation,a,lmin,lmax)
    if pXX == 'p10':
        lacu_path = simu_path + r'\\lacunarity1D'
    else:
        lacu_path = simu_path + r'\\lacunarity3D'
    os.chdir(lacu_path)
    s_tot=[]
    pXX_tot=[]
    for lacu_filename in os.listdir(lacu_path):
        if os.path.isfile(lacu_filename)==False:
            continue
        lacu_file = open(lacu_filename,'r')
        lines = lacu_file.readlines() # title
        for i in range(1,len(lines)):
            line = lines[i]
            if pXX == 'p10':
                pXX_tot.append(float(line.split('\t')[1]))
            if pXX == 'p30':
                pXX_tot.append(float(line.split('\t')[1]))
            elif pXX == 'p32':
                pXX_tot.append(float(line.split('\t')[5]))
            elif pXX == 'percolation':
                pXX_tot.append(float(line.split('\t')[9]))
    pXX_tot=np.asarray(pXX_tot)
    pXX_tot_mean= np.mean(pXX_tot)
    return pXX_tot_mean


def get_lacunarity_curve(root,pXX,L,p32,orientation,a,lmin,lmax):
    simu_path = name_simulation_folder(root,L,p32,orientation,a,lmin,lmax)
    if pXX == 'p10':
        lacu_path = simu_path + r'\\lacunarity1D'
    else:
        lacu_path = simu_path + r'\\lacunarity3D'
    os.chdir(lacu_path)
    s_tot=[]
    pXXLacu_tot=[]
    for lacu_filename in os.listdir(lacu_path):
        if os.path.isfile(lacu_filename)==False:
            continue
        lacu_file = open(lacu_filename,'r')
        lines = lacu_file.readlines() # title
        s=[]
        pXXLacu=[]
        for i in range(1,len(lines)):
            line = lines[i]
            s.append(float(line.split('\t')[-2])*L)
            if pXX == 'p10':
                value = line.split('\t')[0]
                if value != '-nan(ind)':
                    pXXLacu.append(float(value))
                else:
                    pXXLacu.append(np.nan)
            if pXX == 'p30':
                value = line.split('\t')[0]
                if value != '-nan(ind)':
                    pXXLacu.append(float(value))
                else:
                    pXXLacu.append(np.nan)
            elif pXX == 'p32':
                value = line.split('\t')[4]
                if value != '-nan(ind)':
                    pXXLacu.append(float(value))
                else:
                    pXXLacu.append(np.nan)
            elif pXX == 'percolation':
                value = line.split('\t')[8]
                if value != '-nan(ind)':
                    pXXLacu.append(float(value))
                else:
                    pXXLacu.append(np.nan)
        s_tot.append(s)
        pXXLacu_tot.append(pXXLacu)
    s_tot=np.asarray(s_tot)
    s_mean= np.mean(s_tot,axis=0)
    pXXLacu_tot=np.asarray(pXXLacu_tot)
    pXXLacu_mean= np.mean(pXXLacu_tot,axis=0)
    pXXLacu_std= np.std(pXXLacu_tot,axis=0)
    lacuanrity_curve = [s_mean,pXXLacu_mean,pXXLacu_std]
    return lacuanrity_curve



def pXX_lacunarity_theory(pXX,s,root,L,p32,orientation,a,lmin,lmax):
    
    bs=1.
    bf=math.pi/4
    lSmall = 0.05*L
    lBig = L
    #lMean = round(pow(10,(0.5*(np.log10(lSmall) + np.log10(lBig)))),2)
    p10_mean = get_pXX_mean(root,'p10',L,p32,orientation,a,lmin,lmax)
    p30_mean = get_pXX_mean(root,'p30',L,p32,orientation,a,lmin,lmax)
    p32_mean = get_pXX_mean(root,'p32',L,p32,orientation,a,lmin,lmax) 
    p_mean = get_pXX_mean(root,'percolation',L,p32,orientation,a,lmin,lmax) 

    if pXX=='p10':
        return list(map(lambda x : (1/p10_mean)*pow(x,-1), s))

    if pXX=='p30':
        if (a==0):
            if (lmin==lmax):
                l=lmin
                s1 = [elt for elt in s if elt <= l]
                list1 = list(map(lambda x : (bs*bs/p32_mean)*pow(x,-1), s1))
                s2 = [elt for elt in s if elt > l]
                list2 = list(map(lambda x : (1/p30_mean)*pow(x,-3), s2))
                return (list1+list2)

        else:
            size_distribution = get_size_distribution(root,L,p32,orientation,a,lmin,lmax)
            lmin_b = np.nanmin(size_distribution[0])
            lmax_b = np.nanmax(size_distribution[0])
            a, alpha = get_size_distribution_fit(root,L,p32,orientation,a,lmin,lmax)
            a = abs(a)
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            if (lmax<=L):
                f1 = sympy.lambdify(x, sympy.integrate(pow(y,-a),(y,lmin,x)))
                f2 = sympy.lambdify(x, sympy.integrate(pow(y,-a-2),(y,x,lmax)))
                f3 = sympy.lambdify(x, sympy.integrate(pow(y,-a-2),(y,lmin,lmax)))
                s0 = [elt for elt in s if elt<=lmin]
                list0 = list(map(lambda x : (bs*bs/bf)*(alpha/(p30_mean*p30_mean))*pow(x,-1)*f3(x), s0))
                s1 = [elt for elt in s if elt <= lmax and elt>lmin]
                list1 = list(map(lambda x : alpha/(p30_mean*p30_mean) *( pow(x,-3)*f1(x) + (bs*bs/bf)*pow(x,-1)*f2(x) ), s1))
                s2 = [elt for elt in s if elt > lmax]
                list2 = list(map(lambda x : (1/p30_mean)*pow(x,-3), s2))
                return (list0+list1+list2)
            else:
                f1 = sympy.lambdify(x, sympy.integrate(pow(y,-a),(y,lmin,x)))
                f2 = sympy.lambdify(x, sympy.integrate(pow(y,-a-2),(y,x,L)))
                f3 = sympy.lambdify(x, sympy.integrate(pow(y,-a),(y,L,lmax)))
                return list(map(lambda x : alpha/(p30_mean*p30_mean) *( pow(x,-3)*f1(x) + (bs*bs/bf)*pow(x,-1)*f2(x) + bs/(L*L)*pow(x,-1)*f3(x) ), s))
         

    if pXX=='p32':
        if (a==0):
            if (lmin==lmax):
                l=lmin
                s1 = [elt for elt in s if elt <= l]
                list1 = list(map(lambda x : (bs*bs/p32_mean)*pow(x,-1), s1))
                s2 = [elt for elt in s if elt > l]
                list2 = list(map(lambda x : (1/p30_mean)*pow(x,-3), s2))
                return (list1+list2)

        # Powerlaw size
        else:
            size_distribution = get_size_distribution(root,L,p32,orientation,a,lmin,lmax)
            lmin_b = np.nanmin(size_distribution[0])
            lmax_b = np.nanmax(size_distribution[0])
            a, alpha = get_size_distribution_fit(root,L,p32,orientation,a,lmin,lmax)
            a = abs(a)

            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            if (lmax <= L):
                f1 = sympy.lambdify(x, sympy.integrate(pow(y,(4-a)),(y,lmin,x)))
                f2 = sympy.lambdify(x, sympy.integrate(pow(y,(2-a)),(y,x,lmax)))
                f3 = sympy.lambdify(x, sympy.integrate(pow(y,(4-a)),(y,lmin,lmax)))
                s0 = [elt for elt in s if elt<=lmin]
                list0 = list(map(lambda x : (bs*bs/p32_mean)*pow(x,-1), s0))
                s1 = [elt for elt in s if elt <= lmax and elt>lmin]
                list1 = list(map(lambda x : (alpha/(p32_mean*p32_mean))*( bf*bf*pow(x,-3)*f1(x) + bs*bs*bf*(1/x)*f2(x) ), s1 ))
                s2 = [elt for elt in s if elt > lmax]
                list2 = list(map(lambda x : (alpha/(p32_mean*p32_mean))*bf*bf*pow(x,-3)*f3(x), s2))
                return (list0+list1+list2)
            else:
                f1 = sympy.lambdify(x, sympy.integrate(pow(y,(4-a)),(y,lmin,x)))
                f2 = sympy.lambdify(x, sympy.integrate(pow(y,(2-a)),(y,x,L)))
                f3 = sympy.lambdify(x, sympy.integrate(pow(y,-a),(y,L,lmax)))
                return list(map(lambda x : (alpha/(p32_mean*p32_mean))*( bf*bf*pow(x,-3)*f1(x) + bs*bs*bf*(1/x)*f2(x) + bs*bs*bs*L*L*(1/x)*f3(x) ), s ))

    if pXX=='percolation':
        if (a==0):
            if (lmin==lmax):
                l=lmin
                s1 = [elt for elt in s if elt <= l]
                list1 = list(map(lambda x : (bs*bs/p32_mean)*pow(x,-1), s1))
                #list1 = list(map(lambda x : ((math.pi*math.pi) /8) * (p32_mean*p32_mean)/(p_mean*p30_mean) * (bs*bs*bs*bs)/(bf*bf*bf)*pow(x,-1), s1))
                s2 = [elt for elt in s if elt > l]
                list2 = list(map(lambda x : (1/p30_mean)*pow(x,-3), s2))
                return (list1+list2)

        # Powerlaw size
        else:
            size_distribution = get_size_distribution(root,L,p32,orientation,a,lmin,lmax)
            lmin_b = np.nanmin(size_distribution[0])
            lmax_b = np.nanmax(size_distribution[0])
            a, alpha = get_size_distribution_fit(root,L,p32,orientation,a,lmin,lmax)
            a = abs(a)
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            if (lmax <= L):
                f1 = sympy.lambdify(x, sympy.integrate(pow(y,(6-a)),(y,lmin,x)))
                f2 = sympy.lambdify(x, sympy.integrate(pow(y,(4-a)),(y,x,lmax)))
                f3 = sympy.lambdify(x, sympy.integrate(pow(y,(6-a)),(y,lmin,lmax)))
                f4 = sympy.lambdify(x, sympy.integrate(pow(y,(4-a)),(y,lmin,lmax)))
                s0 = [elt for elt in s if elt<=lmin]
                list0 = list(map(lambda x : (alpha*(pow(math.pi,4)/64)/(p_mean*p_mean))*(bs*bs/bf)*pow(x,-1)*f4(x), s0))
                s1 = [elt for elt in s if elt <= lmax and elt>lmin]
                list1 = list(map(lambda x : (alpha/(p_mean*p_mean))*( pow(x,-3)*f1(x) + pow(bs,4)/pow(bf,3)*(1/x)*f2(x) ), s1 ))
                s2 = [elt for elt in s if elt > lmax]
                list2 = list(map(lambda x : (alpha*(pow(math.pi,4)/64)/(p_mean*p_mean))*pow(x,-3)*f3(x), s2))
                return (list0+list1+list2)
            else:
                f1 = sympy.lambdify(x, sympy.integrate(pow(y,(6-a)),(y,lmin,x)))
                f2 = sympy.lambdify(x, sympy.integrate(pow(y,(4-a)),(y,x,L)))
                f3 = sympy.lambdify(x, sympy.integrate(pow(y,-a),(y,L,lmax)))
                return list(map(lambda x : (alpha*(pow(math.pi,4)/64)/(p_mean*p_mean))*( pow(x,-3)*f1(x) + pow(bs,4)/pow(bf,3)*(1/x)*f2(x) ), s ))
                


###################### MAIN ##############################


if __name__ == "__main__":

     ############################### Define parameter #####################################

    nbNetworks = 50
    # We need to decompose computation to manage processors
    nbList=10
    n = int(nbNetworks/nbList)
    sListList=[]
    for i in range(0,nbList):
        sList = [s for s in range(i*n,(i+1)*n)]
        sListList.append(sList)

    p32List = [0.1,0.2,0.3,0.4,0.5]  # fracture intensities
    lmin = 1 # min fracture length
    lmax = 100 # max fracture length
    smin = 2 # minimum investigation scale
    systemSizeList = [400] # system size
    exponentList = [2,3,4,5] # exponents for power law size distributions
    orientationList = [ (np.nan,np.nan,np.nan), (45,135,15)] # orientations
    nbThreads = 96 # nb of threads


    root =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ############################### Create networks #####################################

    print("Generation") 
    t1 = time.perf_counter()
    for sList in sListList:
        generate_networks(root,sList,systemSizeList,p32List,orientationList,exponentList,lmin,lmax,nbThreads)
    t2 = time.perf_counter()
    print("Generation time : {}".format((t2-t1)))

    ############################ Analysis #####################################

    print("Size distribution analysis") 
    t2 = time.perf_counter()
    for sList in sListList:
        launch_size_distribution_analysis_on_networks(root,systemSizeList,p32List,orientationList,exponentList,sList,lmin,lmax,nbThreads,25)
    t3 = time.perf_counter()
    print("Size Distribution analysis time : {}".format((t3-t2)))

    print("Lacunarity analysis") 
    t3 = time.perf_counter()
    for sList in sListList:
        print(sList)
        launch_lacunarity_analysis_on_networks(root,systemSizeList,p32List,orientationList,exponentList,sList,lmin,lmax,nbThreads,smin)  
    t4 = time.perf_counter()
    print("Lacunarity analysis time : {}".format((t4-t3)))


################################################################################################

    



