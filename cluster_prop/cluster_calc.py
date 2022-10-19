import math
import numpy as np
from ase import Atoms
from ase.io import read
from ase.build.tools import sort
from itertools import combinations

original_cell = read('../data/Li_640.xyz')
li = sort(original_cell, tags=original_cell.positions[:, 2])
_surface_grid = li.positions[:64][:, :2]
a=_surface_grid[1,1]-_surface_grid[0,1]
_surface_grid = np.reshape(_surface_grid,(8,8,2))
#
def find_vac_z(atom_object):
    """

    :param atom_object: ASE Atoms object
    :return: the location of vacuum layer along z-axis
    """
    all_z = np.unique(atom_object.positions[:, 2])
    for z in all_z:
        counter = 0
        for atom in atom_object.positions:
            if atom[2] == z:
                counter += 1
        # print(z,counter)
        if counter != 64:
            return z

def vac_pos(surface_pos):
    """

    :param surface_pos:the vacuum layer
    :return: vacancy position
    """
    grid=np.zeros((8,8))
    surface_pos_=enumerate(surface_pos)
    origin=surface_pos[0]
    _,cur=next(surface_pos_)
    counter=0
    for _ in range(64):
        i = counter // 8
        j = counter % 8

        if (cur[0]-(origin[0]+i*a))<1e-4:
            if(cur[1]-(origin[1]+j*a))<1e-4:
                grid[i,j]=1
                try:
                    _,cur=next(surface_pos_)
                except StopIteration:
                    break

        counter+=1
    pos_list=[]
    for i,row in enumerate(grid):
        for j,index in enumerate(row):
            if(index==0):
                pos_list.append(_surface_grid[i,j])
    return np.array(pos_list)

def defect_pos(system):
    """

    :param system:
    :return:
    """
    pos_list=[]
    for i in range(len(system)):
        if system.get_atomic_numbers()[i] != 3:
            #surface_z = system.positions[i][2]
            pos_list.append(system.positions[i][:2])
    return np.array(pos_list)


def find_surface(system):
    surface_z = 0
    surface_atoms = []
    for i in range(len(system)):
        if system.get_atomic_numbers()[i] != 3:
            surface_z = system.positions[i][2]
    for i in system:
        if i.z >= surface_z:
            surface_atoms.append(i)
    return Atoms(surface_atoms, pbc=True)

def find_defect_surface(system):
    surface_z = 0
    surface_atoms = []
    for i in range(len(system)):
        if system.get_atomic_numbers()[i] != 3:
            surface_z = system.positions[i][2]
    #    if i.z >= surface_z and i.:
    #        surface_atoms.append(i)
    return Atoms(surface_atoms, pbc=True)

def find_surface_atom_pos(atom_object):
    unique_atom=set(atom_object.get_chemical_symbols())
    if len(unique_atom)==1:
        z = find_vac_z(atom_object)
        mask = (atom_object.positions[:, 2] == z)
        return vac_pos(atom_object.positions[mask, :])
    elif len(unique_atom)>1:
        return defect_pos(atom_object)



def circle_radius(atom_pos):
    n = len(atom_pos)
    centroid = np.mean(atom_pos, axis=0)
    r_sqr = np.max(np.linalg.norm((atom_pos - centroid), ord=2))
    r_std = np.mean(np.linalg.norm((atom_pos-centroid),ord=2))
    radius = np.sqrt(r_sqr)

    return radius,r_std


def circle_from_3_points(defect:list) -> tuple[list, float]:
    a,b,c=defect[0],defect[1],defect[2]
    z1=complex(a[0],a[1])
    z2=complex(b[0],b[1])
    z3=complex(c[0],c[1])

    if (z1 == z2) or (z2 == z3) or (z3 == z1):
        raise ValueError(f'Duplicate points: {z1}, {z2}, {z3}')

    w = (z3 - z1) / (z2 - z1)

    # You should probably use `math.isclose(w.imag, 0)` for floating point comparisons
    if math.isclose(w.imag, 0):
        raise ValueError(f'Points are collinear: {z1}, {z2}, {z3}')

    c = (z2 - z1) * (w - abs(w) ** 2) / (2j * w.imag) + z1  # Simplified denominator
    r = abs(z1 - c)

    return [c.real,c.imag], r



def find_maxr(defect):
    comb=combinations(defect,3)
    maxr=-float("inf")
    for defect in comb:
        _,r=circle_from_3_points(defect)
        if r>maxr:
            maxr=r
    return maxr
