"""Rotatable ball-and-stick views of predicted substitution pathways."""
import math
import time
from trajectory import Trajectory
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


class ReactionView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.rows = []
        self.choice = ttk.Combobox(self, state="readonly", width=38)
        self.choice.pack()
        self.choice.bind("<<ComboboxSelected>>", lambda _: self.refresh())
        self.timer = None
        self.track = None
        self.position = tk.DoubleVar(value=0)
        self.clouds = tk.BooleanVar(value=True)
        self.contacts = tk.BooleanVar(value=True)
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.play = ttk.Button(bar, text="Play", command=self.toggle)
        self.play.pack(side="left")
        self.slider = ttk.Scale(bar, from_=0, to=1, variable=self.position,
                                command=lambda _: self.seek())
        self.slider.pack(side="left", expand=True, fill="x", padx=6)
        self.slider.bind("<ButtonPress-1>", lambda _: self.pause())
        self.slider.bind("<KeyPress>", lambda _: self.pause())
        self.percent = ttk.Label(bar, width=5)
        self.percent.pack(side="left")
        options = ttk.Frame(self)
        options.pack()
        for label, value in (("Electron clouds", self.clouds), ("Close contacts", self.contacts)):
            ttk.Checkbutton(options, text=label, variable=value, command=self.draw).pack(side="left")
        self.bind("<Unmap>", lambda _: self.pause())
        self.canvas = tk.Canvas(self, width=490, height=230, background="#f6f8fc")
        self.canvas.pack(fill="both", expand=True)
        self.caption = ttk.Label(self, wraplength=480)
        self.caption.pack()
        self.reason = ttk.Label(self, wraplength=480)
        self.reason.pack()
        ttk.Label(self, wraplength=480, text="Schematic timeline | Drag to rotate. Blue halos show approximate electron-cloud extent. Red lines mark possible steric repulsion.").pack()
        self.canvas.bind("<ButtonPress-1>", self.start)
        self.canvas.bind("<B1-Motion>", self.rotate)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.scene = None
        self.cache = {}

    def set_prediction(self, substrate, nuc, leaving, rows, labels, context=""):
        self.context = context
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

    def pause(self):
        if self.timer is not None:
            self.after_cancel(self.timer)
            self.timer = None
        self.play.configure(text="Play")

    def toggle(self):
        if self.timer is not None:
            self.pause()
        elif self.track is not None:
            if self.position.get() >= 1:
                self.position.set(0)
            self.started = time.monotonic() - self.position.get()*8
            self.play.configure(text="Pause")
            self.tick()

    def tick(self):
        self.timer = None
        self.position.set(min(1, (time.monotonic()-self.started)/8))
        self.seek()
        if self.position.get() < 1:
            self.timer = self.after(33, self.tick)
        else:
            self.pause()

    def seek(self):
        t = self.position.get()
        self.percent.configure(text=f"{t:.0%}")
        if self.track:
            *self.scene, note = self.track.frame(t)
            self.caption.configure(text=note)
        self.draw()

    def refresh(self):
        self.pause()
        self.scene = self.track = None
        self.position.set(0)
        self.yaw, self.pitch = 0.4, -0.3
        index = self.choice.current()
        note = "Run a prediction to see a model."
        reason = ""
        if self.rows and index >= 0:
            product, path, _ = self.rows[index]
            try:
                if path.startswith("E"):
                    self.scene = (*model(product), [], [])
                    note = "Elimination product."
                else:
                    if index not in self.cache:
                        self.cache[index] = Trajectory(self.substrate, self.nuc, product,
                                                       self.leaving, path, model)
                    self.track = self.cache[index]
                    n = self.track.crowding
                    reason = (f"Carbon neighbors at the reaction center: {n}. "
                              + ("More alkyl substitution generally stabilizes a carbocation. "
                                 if self.track.sn1 else
                                 "More alkyl substitution generally crowds the backside approach. ")
                              + self.context)
            except (RuntimeError, ValueError, IndexError, StopIteration):
                note = "3D geometry unavailable for this structure."
        state = "normal" if self.track else "disabled"
        self.play.configure(state=state)
        self.slider.configure(state=state)
        self.reason.configure(text=reason)
        self.caption.configure(text=note)
        self.seek()

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        if self.scene is None:
            return
        atoms, coords, bonds, partial, near = self.scene
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
        if self.clouds.get():
            for i in depth.argsort():
                x,y = screen[i]
                symbol = atoms[i].rstrip("+-")
                radius = Chem.GetPeriodicTable().GetRvdw(symbol)*scale
                canvas.create_oval(x-radius,y-radius,x+radius,y+radius,
                                   fill="#97c1e5", stipple="gray25", outline="#c2d9ed")
        for i, j in bonds:
            canvas.create_line(*screen[i], *screen[j], width=3, fill="#8493a5")
        for i,j,order in partial:
            canvas.create_line(*screen[i], *screen[j], width=1+3*order, dash=(4,3), fill="#914a00")
        if self.contacts.get():
            for i,j in near:
                canvas.create_line(*screen[i], *screen[j], dash=(2,3), fill="#b82431", width=2)
        for i in depth.argsort():
            x, y = screen[i]
            symbol = atoms[i]
            r = 5 if symbol == "H" else 10
            canvas.create_oval(x-r, y-r, x+r, y+r, fill=colors.get(symbol.rstrip("+-"), "#666"), outline="#445")
            if symbol != "H":
                canvas.create_text(x, y, text=symbol, fill="white", font=("TkDefaultFont", 8))
