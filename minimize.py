import MDAnalysis as mda
from rdkit import Chem
import subprocess
from rdkit.Chem.rdchem import AtomPDBResidueInfo
from rdkit.Chem import rdmolops
from rdkit.Chem import ChemicalForceFields
from openmm import app
from rdkit.Geometry import Point3D
import os
import argparse

HDONER = "[$([O,S;+0]),$([N;$(Na),$(NC=[O,S]);H2]),$([N;$(N[S,P]=O)]);!H0]"
UNWANTED_H = "[#1;$([#1][N;+1;H2]),$([#1][N;!H2]a)]"

def parse_arguments():
    parser = argparse.ArgumentParser(description="minimize the polar hydrogens in vina docking")
    
    parser.add_argument("-p", "--pdb", required=True, help="Path to pdb file.")
    parser.add_argument("-l", "--ligfile", required=True, help="Path to ligand sdf file.")
    parser.add_argument("-o", "--outfile", required=True, help="Path to output file.")
    
    args = parser.parse_args()
    return args

pdb = parse_arguments().pdb
ligfile = parse_arguments().ligfile
outfile = parse_arguments().outfile


def convert_to_pdb(sdf):
    cmd = f"obabel {sdf} -O {sdf}.pdb"
    subprocess.call(cmd, shell=True)

def constrain_minimize(mol, constrain_list):
    ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ff_property, confId=0, ignoreInterfragInteractions=False)

    for query_atom_idx in constrain_list:
        ff.MMFFAddPositionConstraint(query_atom_idx, 0.0, 1000)

    ff.Initialize()

    max_minimize_iteration = 10
    for _ in range(max_minimize_iteration):
        minimize_seed = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        if minimize_seed == 0:
            break
    
    return mol


ligand = Chem.SDMolSupplier(ligfile, removeHs=False)[0]
pattern = Chem.MolFromSmarts(HDONER)
match = ligand.HasSubstructMatch(pattern)

if match == False:
    cmd = f'cp {ligfile} {outfile}'
    subprocess.call(cmd, shell=True)
    exit()

else:
    pdbfile = app.PDBFile(pdb)
    protein_universe = mda.Universe(pdbfile)
    convert_to_pdb(ligfile)
    ligand_universe = mda.Universe(ligfile+'.pdb')
    merge_pdb = mda.Merge(protein_universe.atoms, ligand_universe.atoms)
    pro_pocket = merge_pdb.select_atoms('byres protein and around 4.0 (resname UNL)')
    mda_to_rdkit = mda._CONVERTERS['RDKIT']().convert
    pro_pocket_mol = mda_to_rdkit(pro_pocket)

    # set residue name
    for atom in pro_pocket_mol.GetAtoms():
        atom.GetMonomerInfo().SetResidueName("PRO")

    for atom in ligand.GetAtoms():
        monomer_info = atom.GetMonomerInfo()

        # If there is no monomer information, create it
        if monomer_info is None:
            # Create a new residue info object with the desired residue name
            residue_info = AtomPDBResidueInfo()
            residue_info.SetResidueName("LIG")
            atom.SetMonomerInfo(residue_info)
        else:
            monomer_info.SetResidueName("LIG")

    # combine ligand and protein pocket
    complex = rdmolops.CombineMols(ligand, pro_pocket_mol)

    # get atom index list to be constrained
    constrain_list = []
    unwanted_H_pattern = Chem.MolFromSmarts(UNWANTED_H)
    unwanted_H_atom_idx_list = list(complex.GetSubstructMatches(unwanted_H_pattern))
    unwanted_H_atom_idx_list = [unwanted_H_atom_idx_tuple[0] for unwanted_H_atom_idx_tuple in unwanted_H_atom_idx_list]

    for atom in complex.GetAtoms():
        if atom.GetMonomerInfo().GetResidueName() == "PRO":
            constrain_list.append(atom.GetIdx())
        if atom.GetMonomerInfo().GetResidueName() == "LIG":
            if atom.GetSymbol() != "H":
                constrain_list.append(atom.GetIdx())
            if atom.GetIdx() in unwanted_H_atom_idx_list:
                constrain_list.append(atom.GetIdx())

    complex_min = constrain_minimize(complex, constrain_list)

    #assign minimized conformer to ligand
    coord_dict = {}
    for atom in complex_min.GetAtoms():
        if atom.GetMonomerInfo().GetResidueName() == "LIG":
            coord_dict[atom.GetIdx()] = complex_min.GetConformer().GetAtomPosition(atom.GetIdx())
            
    lig_conf = Chem.Conformer()
    for idx in range(len(ligand.GetAtoms())):
        atom_coords = coord_dict[idx]
        atom_coords_point_3D = Point3D(atom_coords[0], atom_coords[1], atom_coords[2])
        lig_conf.SetAtomPosition(idx, atom_coords_point_3D)

    ligand.RemoveAllConformers()
    ligand.AddConformer(lig_conf)

    writer = Chem.SDWriter(outfile)
    writer.write(ligand)
    writer.close()

    os.remove(ligfile+'.pdb')
