#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:53:16 2020

@author: shanyang

"""

import numpy as np
import numpy.linalg as lg
import os
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'Times','size' : 16}

matplotlib.rc('font', **font)

def read_fc3(path):
    try:
        file = os.path.join(path,'outfile.forceconstant_thirdorder')
        with open(file) as fc3:
            fc3_raw = fc3.readlines()
    except FileNotFoundError as notexist:
        print('File do no exist {}'.format(notexist))
    finally:
        return fc3_raw


def read_poscar(path):
    fract = []
    try:
        file = os.path.join(path,'POSCAR')
        with open(file) as poscar:
            poscar = poscar.readlines()
            a = poscar[2];
            b = poscar[3];
            c = poscar[4];
            abc =np.asarray(a.strip('/n').split(), dtype= float),  np.asarray(b.strip('/n').split(), dtype= float),  np.asarray(c.strip('/n').split(), dtype= float)
            total_atoms = np.array(poscar[6].strip('/n').split(), dtype= int).sum()
            for item in range(8,8+total_atoms):
                fract.append(np.array(poscar[item].strip('/n').split(), dtype= float))
            return abc, fract
        
    except FileNotFoundError as notexist:
        print('File do no exist {}'.format(notexist))
    

def fc3_atom(start,fc3_raw,fract):
    triples = int(fc3_raw[start+1].strip('\n').split()[0])
    end = start+1+triples*15;
    #print(f"triples {triples} for atom, end {end} index")

    x_posi = fc3_raw[start+2:start+2+triples*15:15];
    y_posi = fc3_raw[start+3:start+3+triples*15:15];
    z_posi = fc3_raw[start+4:start+4+triples*15:15];
    # atoms positions (triples index, abc axis, xyz in abc axis)
    atom_positions = np.zeros((triples,3,3))
        
    x_idx = start+8;
    y_idx = start+9;
    z_idx = start+10;

    triples_x = fc3_raw[x_idx:x_idx+triples*15:15]
    triples_y = fc3_raw[y_idx:y_idx+triples*15:15]
    triples_z = fc3_raw[z_idx:z_idx+triples*15:15]
    atom = np.zeros((triples,3,3))
    for i in range(triples):
        atom[i,0,:] = np.asarray(triples_x[i].strip('\n').split(),dtype = float);
        atom[i,1,:] = np.asarray(triples_y[i].strip('\n').split(),dtype = float);
        atom[i,2,:] = np.asarray(triples_z[i].strip('\n').split(),dtype = float);
        atom_positions[i,0,:] = fract[np.asarray(x_posi[i].strip('\n').split()[0],dtype = int)-1]
        atom_positions[i,1,:] = fract[np.asarray(y_posi[i].strip('\n').split()[0],dtype = int)-1]
        atom_positions[i,2,:] = fract[np.asarray(z_posi[i].strip('\n').split()[0],dtype = int)-1]
    return atom, end, atom_positions

def fc3_all_atoms(fc3_raw,fract):
    start = 1;   
    atoms = []
    positions = []
    while (start < len(fc3_raw)-1):
        each_atom,each_end,each_position= fc3_atom(start,fc3_raw,fract)
        atoms.append(each_atom)
        positions.append(each_position)
        start = each_end;
    return atoms, positions

def normfc3_all_atoms(atoms_fc3,abc,positions_fc3):
    atoms_num = len(atoms_fc3)
    normfc3 = []
    distancefc3= []
    for atom in range(atoms_num):
        each_distance = []
        each_fc3=[]
        for item in range(len(atoms_fc3[atom])):
            each_fc3.append(lg.norm(atoms_fc3[atom][item]))
            dist1 = np.matmul(np.transpose(abc),positions_fc3[atom][item][1]-positions_fc3[atom][item][0])
            dist2 = np.matmul(np.transpose(abc),positions_fc3[atom][item][2]-positions_fc3[atom][item][0])
            #each_distance.append(min(lg.norm(dist1),lg.norm(dist2)))
            each_distance.append(lg.norm(dist1))
            #each_distance.append(lg.norm(dist1)+lg.norm(dist2))
            #print(f"index {item} of triple and fc {lg.norm(atoms_fc3[0][item])}");
        normfc3.append(each_fc3)
        distancefc3.append(each_distance)
    return normfc3, distancefc3

def normfc3_plot(atoms,normfc3,distancefc3,path,alpha=0.8):
    #atoms = len(normfc3)
    fig, ax = plt.subplots(figsize=(10,8))
    for atom in range(atoms):
        ax.plot(distancefc3[atom],np.asarray(normfc3[atom]),'o',label = f"atom {atom}",alpha =alpha)
    ax.legend(shadow = None, frameon = False,fancybox = None)
    ax.set_xlim([0.5,5])
    ax.set_ylim([0,60])
    ax.set_xlabel('Triple distance (A)')
    ax.set_ylabel('Norm of third order forceconstant (eV/A^3)')
    fig.savefig(path+'/fc3.png',dpi=150)
    
    

r_path = '/Users/shanyang/Desktop/VO2_temperature/VO2_R/425K/MD-5'
fc3_raw=read_fc3(r_path)
# lattice constant 
abc,fract= read_poscar(r_path)
atoms_fc3, positions_fc3 = fc3_all_atoms(fc3_raw,fract)
normfc3_r,distancefc3_r = normfc3_all_atoms(atoms_fc3,abc,positions_fc3)
#normfc3_plot(6,normfc3_r,distancefc3_r,r_path)


# M1 /Users/shanyang/Desktop/VO2_temperature/VO2_M1/M1-300K/MD-1
m1_path = "/Users/shanyang/Desktop/VO2_temperature/VO2_M1/M1-300K/MD-6"
fc3_raw=read_fc3(m1_path)
# lattice constant 
abc,fract= read_poscar(m1_path)
atoms_fc3, positions_fc3 = fc3_all_atoms(fc3_raw,fract)
normfc3_m1,distancefc3_m1 = normfc3_all_atoms(atoms_fc3,abc,positions_fc3)
#normfc3_plot(12,normfc3_m1,distancefc3_m1,m1_path)


fc_m1 = np.loadtxt('/Users/shanyang/Desktop/VO2_temperature/VO2_M1/M1-300K/MD-6/fc_plot')
fc_r = np.loadtxt('/Users/shanyang/Desktop/VO2_temperature/VO2_R/425K/MD-5/fc_plot')
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(fc_m1[:,1],fc_m1[:,2],'o',label = "M1",color = 'blue')
ax.plot(fc_r[:,1],fc_r[:,2],'o',label = "Rutile",color = 'red')
ax.legend(shadow = None, frameon = False,fancybox = None)
ax.set_xlim([1,5])
ax.set_ylim([0,12])
ax.set_xlabel(r'Pair distance $(A)$')
ax.set_ylabel(r'Norm of second order forceconstant $(eV/A^{2})$')
fig.savefig(m1_path+'/fc2_R_M1.png',dpi=150)


fig, ax = plt.subplots(figsize=(8,5))
for atom in range(2):
    ax.plot(distancefc3_r[atom],np.asarray(normfc3_r[atom]),'o',color='red',alpha =0.8)
for atom in range(2):
    ax.plot(distancefc3_m1[atom],np.asarray(normfc3_m1[atom]),'o',color='blue',alpha =0.8)   
        
ax.legend(shadow = None, frameon = False,fancybox = None)
ax.set_xlim([1,5])
ax.set_ylim([0,60])
ax.set_xlabel(r'Triple distance $(A)$')
ax.set_ylabel(r'Norm of third order forceconstant $(eV/A^{3})$')
fig.savefig(m1_path+'/fc3.png',dpi=150)


#ax.legend(shadow = None, frameon = False,fancybox = None)
ax.set_xlim([1,5])
ax.set_ylim([0,60])
ax.set_xlabel(r'Triple distance $(A)$')
ax.set_ylabel(r'Norm of third order forceconstant $(eV/A^{3})$')
#fig.savefig(m1_path+'/fc3.png',dpi=150)







    






        