"""Rotatable ball-and-stick views of predicted substitution pathways."""
import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def model(mol):
    if mol is None:
        raise ValueError("Invalid structure")
    mol = Chem.AddHs(Chem.Mol(mol))
    settings = AllChem.ETKDGv3()
    settings.randomSeed = 182
    if AllChem.EmbedMolecule(mol, settings) < 0:
        settings.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, settings) < 0:
            raise ValueError("Conformer generation failed")
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return atoms, mol.GetConformer().GetPositions(), bonds


def attack(substrate, nuc, product, leaving, sn1):
    # The predictor appends the new core-to-nucleophile bond last.
    formed = product.GetBondWithIdx(product.GetNumBonds() - 1)
    target = formed.GetBeginAtomIdx()
    core_size = product.GetNumAtoms() - nuc.GetNumAtoms()
    donor = formed.GetEndAtomIdx() - core_size
    if sn1:
        core = Chem.RWMol(product)
        for index in reversed(range(core_size, product.GetNumAtoms())):
            core.RemoveAtom(index)
        center = core.GetAtomWithIdx(target)
        center.SetFormalCharge(1)
        center.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.SanitizeMol(core)
        atoms, points, bonds = model(core)
        neighbors = [j if i == target else i for i, j in bonds if target in (i, j)]
        if len(neighbors) != 3:
            raise ValueError("Unsupported cation geometry")
        a, b, c = points[neighbors]
        direction = np.cross(b - a, c - a)
        atoms[target] += "+"
    else:
        target += int(target >= leaving)
        atoms, points, bonds = model(substrate)
        direction = points[target] - points[leaving]
    direction /= np.linalg.norm(direction)
    incoming, coords, links = model(nuc)
    coords -= coords[donor].copy()
    # Orient the nucleophile's other atoms away from the substrate.
    if coords.mean(axis=0) @ direction < 0:
        axis = np.eye(3)[np.argmin(abs(direction))]
        axis = np.cross(direction, axis)
        axis /= np.linalg.norm(axis)
        coords = 2 * np.outer(coords @ axis, axis) - coords
    coords += points[target] + 3 * direction
    offset = len(atoms)
    return (atoms + incoming, np.vstack((points, coords)),
            bonds + [(offset+i, offset+j) for i, j in links], (target, offset+donor))


class ReactionView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.rows = []
        self.choice = ttk.Combobox(self, state="readonly", width=38)
        self.choice.pack()
        self.choice.bind("<<ComboboxSelected>>", lambda _: self.refresh())
        self.stage = tk.StringVar(value="Attack")
        bar = ttk.Frame(self)
        bar.pack()
        for stage in ("Reactant", "Attack", "Product"):
            ttk.Radiobutton(bar, text=stage, variable=self.stage, value=stage,
                            command=self.refresh).pack(side="left")
        self.canvas = tk.Canvas(self, width=490, height=230, background="#f6f8fc")
        self.canvas.pack(fill="both", expand=True)
        self.caption = ttk.Label(self, wraplength=480)
        self.caption.pack()
        self.canvas.bind("<ButtonPress-1>", self.start)
        self.canvas.bind("<B1-Motion>", self.rotate)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.scene = None
        self.cache = {}

    def set_prediction(self, substrate, nuc, leaving, rows, labels):
        self.substrate, self.nuc, self.leaving, self.rows = substrate, nuc, leaving, rows
        self.cache.clear()
        self.choice["values"] = labels
        self.choice.set("")
        if rows:
            self.choice.current(0)
        self.refresh()

    def start(self, event):
        self.pointer = event.x, event.y

    def rotate(self, event):
        x, y = self.pointer
        self.yaw += (event.x - x) / 90
        self.pitch += (event.y - y) / 90
        self.start(event)
        self.draw()

    def refresh(self):
        self.scene = None
        self.yaw, self.pitch = 0.4, -0.3
        stage, index = self.stage.get(), self.choice.current()
        note = "Run a prediction to see a model."
        if self.rows:
            product, path, _ = self.rows[index]
            note = "Drag to rotate."
            try:
                key = index, stage
                if stage == "Attack" and path.startswith("E"):
                    note = "Elimination pathway. Select Product to inspect the predicted alkene."
                else:
                    if key not in self.cache:
                        if stage == "Attack":
                            self.cache[key] = attack(self.substrate, self.nuc, product,
                                                     self.leaving, path.startswith("SN1"))
                        else:
                            mol = self.substrate if stage == "Reactant" else product
                            self.cache[key] = (*model(mol), None)
                    self.scene = self.cache[key]
                    if stage == "Attack":
                        note += (" SN1: one possible approach to the carbocation."
                                 if path.startswith("SN1") else " SN2: backside approach opposite the leaving group.")
                        note += " Dashed line shows the forming bond."
            except (RuntimeError, ValueError, IndexError):
                note = "3D geometry unavailable for this structure."
        self.caption.configure(text=note)
        self.draw()

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        if self.scene is None:
            return
        atoms, coords, bonds, forming = self.scene
        points = coords - coords.mean(axis=0)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, y, z = points.T
        depth = -sy*x + cy*z
        w, h = canvas.winfo_width(), canvas.winfo_height()
        scale = max(1, min(w, h)-40) / max(2*np.linalg.norm(points, axis=1).max(), 1)
        screen = np.column_stack((w/2 + scale*(cy*x+sy*z), h/2 - scale*(cp*y-sp*depth)))
        depth = sp*y + cp*depth
        colors = {"C":"#485465", "O":"#c83e46", "N":"#335ec1", "Br":"#934023",
                  "Cl":"#377541", "F":"#377541", "I":"#744da2", "H":"#e4e9ee"}
        for i, j in bonds:
            canvas.create_line(*screen[i], *screen[j], width=3, fill="#8493a5")
        if forming:
            i, j = forming
            canvas.create_line(*screen[i], *screen[j], width=2, dash=(4, 3), fill="#914a00")
        for i in depth.argsort():
            x, y = screen[i]
            symbol = atoms[i]
            r = 5 if symbol == "H" else 10
            canvas.create_oval(x-r, y-r, x+r, y+r, fill=colors.get(symbol.rstrip("+"), "#666"), outline="#445")
            if symbol != "H":
                canvas.create_text(x, y, text=symbol, fill="white", font=("TkDefaultFont", 8))
