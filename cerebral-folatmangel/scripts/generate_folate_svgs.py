from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdFMCS


SMILES = {
    "folate": "O=C(c1ccc(NCC2=CN=C3C(=N2)C(=O)NC(=N3)N)cc1)N[C@@H](CCC(=O)[O-])C(=O)[O-]",
    "tetrahydrofolate": "C1[C@@H](NC2=C(N1)N=C(NC2=O)N)CNc3ccc(cc3)C(=O)N[C@@H](CCC(=O)[O-])C(=O)[O-]",
    "mthf_5": "CN1[C@H](CNc2ccc(cc2)C(=O)N[C@@H](CCC(=O)[O-])C(=O)[O-])CNC2=C1C(=O)NC(=N2)N",
    "folinate": "C1C(N(C2=C(N1)N=C(NC2=O)N)C=O)CNc3ccc(cc3)C(=O)N[C@@H](CCC(=O)[O-])C(=O)[O-]",
}

def make_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


def mcs_atom_map(reference: Chem.Mol, target: Chem.Mol) -> list[tuple[int, int]]:
    mcs = rdFMCS.FindMCS(
        [reference, target],
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        maximizeBonds=True,
    )
    query = Chem.MolFromSmarts(mcs.smartsString)
    if query is None:
        raise ValueError("Failed to build MCS query")
    ref_match = reference.GetSubstructMatch(query)
    target_match = target.GetSubstructMatch(query)
    if not ref_match or not target_match:
        raise ValueError("Failed to map target to reference")
    return list(zip(ref_match, target_match))


def align_to_reference(reference: Chem.Mol, target: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(target)
    atom_map = mcs_atom_map(reference, mol)
    rdDepictor.GenerateDepictionMatching2DStructure(
        mol,
        reference,
        atomMap=atom_map,
    )
    return mol


def draw_svg(mol: Chem.Mol, output_file: Path) -> None:
    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    drawer = Draw.MolDraw2DSVG(900, 420)
    options = drawer.drawOptions()
    options.padding = 0.03
    options.bondLineWidth = 2
    options.multipleBondOffset = 0.18
    options.fixedBondLength = 38
    options.minFontSize = 18
    options.maxFontSize = 28
    options.annotationFontScale = 0.9
    options.addStereoAnnotation = False
    options.useBWAtomPalette()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8 and atom.GetFormalCharge() == -1:
            atom.SetFormalCharge(0)
            options.atomLabels[atom.GetIdx()] = "-O"
    Draw.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    output_file.write_text(drawer.GetDrawingText(), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "images" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    folate = make_mol(SMILES["folate"])
    rdDepictor.Compute2DCoords(folate)

    aligned = {"folate": folate}
    for key in ("tetrahydrofolate", "mthf_5", "folinate"):
        aligned[key] = align_to_reference(folate, make_mol(SMILES[key]))

    for key, mol in aligned.items():
        draw_svg(mol, out_dir / f"{key}.svg")


if __name__ == "__main__":
    main()
