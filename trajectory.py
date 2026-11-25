"""Schematic reaction coordinates in the atom order of the starting materials."""
import numpy as np
from rdkit import Chem


def fraction(t, start, end):
    u = min(1., max(0., (t-start)/(end-start)))
    return u*u*(3-2*u)


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def branch(bonds, root, blocked):
    visited = {blocked}
    stack = [root]
    result = []
    while stack:
        i = stack.pop()
        if i in visited:
            continue
        visited.add(i)
        result.append(i)
        stack.extend(b if a == i else a for a, b in bonds if i in (a, b))
    return result


class Trajectory:
    def __init__(self, substrate, nuc, product, leaving, path, embed):
        atoms, points, bonds = embed(substrate)
        incoming, coords, links = embed(nuc)
        self.atoms = atoms + incoming
        self.points, self.bonds = points, bonds
        self.leaving = leaving
        self.center = next(n.GetIdx() for n in substrate.GetAtomWithIdx(leaving).GetNeighbors()
                           if n.GetSymbol() == "C")
        self.departing = branch(bonds, leaving, self.center)
        formed = product.GetBondWithIdx(product.GetNumBonds()-1)
        target = formed.GetBeginAtomIdx()
        self.target = target + int(target >= leaving)
        donor = formed.GetEndAtomIdx() - (product.GetNumAtoms()-nuc.GetNumAtoms())
        self.offset = len(atoms)
        self.donor = self.offset + donor
        self.incoming = coords - coords[donor]
        self.links = [(i+self.offset,j+self.offset) for i,j in links]
        self.sn1 = path.startswith("SN1")
        self.shift = self.target != self.center
        self.group = None
        if self.shift:
            self.group = (product.GetIntProp("migratingAtom") if product.HasProp("migratingAtom")
                          else next(i for i in self.neighbors(self.target) if atoms[i] == "H"))
            self.moving = branch(bonds, self.group, self.target)
        self.axis = unit(points[leaving]-points[self.center])
        self.initial_charge = {a.GetIdx()+offset:a.GetFormalCharge()
                               for m,offset in ((substrate,0),(nuc,self.offset)) for a in m.GetAtoms()}
        self.free_neighbors = [i for i in self.neighbors(self.center) if i != leaving]
        self.crowding = sum(atoms[i]=="C" for i in self.free_neighbors)
        self.radii = np.array([Chem.GetPeriodicTable().GetRvdw(a) for a in self.atoms])

    def neighbors(self, atom):
        return [b if a==atom else a for a,b in self.bonds if atom in (a,b)]

    def frame(self, t):
        pts = self.points.copy()
        c, lg, target = self.center, self.leaving, self.target
        loss = fraction(t, .05, .30) if self.sn1 else fraction(t,.12,.89)
        move = fraction(t,.33,.60) if self.shift else 0
        addition = fraction(t,.64 if self.shift else .45, .95) if self.sn1 else loss
        pts[self.departing] += 3.4*self.axis*loss
        # Move the centre through the plane of its retained substituents.
        if len(self.free_neighbors)==3:
            a,b,d = pts[self.free_neighbors]
            normal = unit(np.cross(b-a,d-a))
            to_plane = normal * ((a-pts[c]) @ normal)
            pts[c] += to_plane * loss * (1 if self.sn1 else 2)
        approach = -self.axis
        if self.shift:
            destination = pts[c] + self.axis*np.linalg.norm(self.points[self.group]-self.points[target])
            pts[self.moving] += move*(destination-self.points[self.group])
            retained = [i for i in self.neighbors(target) if i != self.group]
            if len(retained)==3:
                a,b,d = pts[retained]
                normal = unit(np.cross(b-a,d-a))
                if normal @ (self.points[self.group]-pts[target]) < 0:
                    normal *= -1
                pts[target] += move * normal * ((a-pts[target]) @ normal)
                approach = normal
        if self.sn1 and not self.shift and len(self.free_neighbors)==3:
            approach = normal if normal @ self.axis < 0 else -normal
        if self.sn1:
            pts[target] += .35*addition*approach
        incoming = self.incoming.copy()
        if incoming.mean(axis=0) @ approach < 0:
            axis = unit(np.cross(approach, np.eye(3)[np.argmin(abs(approach))]))
            incoming = 2*np.outer(incoming @ axis, axis)-incoming
        end_distance = (Chem.GetPeriodicTable().GetRcovalent("C") +
                        Chem.GetPeriodicTable().GetRcovalent(self.atoms[self.donor]))
        incoming += pts[target] + approach*(5*(1-addition)+end_distance*addition)
        bonds = list(self.links)
        partial = []
        def changing(i,j,order):
            if order >= .999:
                bonds.append((i,j))
            elif order > .001:
                partial.append((i,j,order))
        for i,j in self.bonds:
            if {i,j} == {lg,c}:
                changing(i,j,1-loss)
            elif self.shift and {i,j} == {self.group,target}:
                changing(i,j,1-move)
            else:
                bonds.append((i,j))
        if self.shift:
            changing(c,self.group,move)
        changing(target,self.donor,addition)
        charge = self.initial_charge.copy()
        if loss > .99:
            charge[lg] -= 1
        if self.sn1 and loss > .99 and addition < .99:
            charge[target if move>.5 else c] = 1
        if addition > .99:
            charge[self.donor] += 1
        labels = [s+({1:"+",-1:"-"}.get(charge.get(i),"")) for i,s in enumerate(self.atoms)]
        near = []
        if .01 < addition < .99:
            for i in range(self.offset):
                if i == target or i in self.departing:
                    continue
                if np.linalg.norm(pts[i]-incoming[self.donor-self.offset]) < .85*(self.radii[i]+self.radii[self.donor]):
                    near.append((i,self.donor))
        if t >= .95:
            note = "Product formed. The new carbon-nucleophile bond is complete."
        elif not self.sn1:
            note = ("SN2 | One concerted step. Backside approach aligns the donating electron pair with the C-LG antibonding orbital. "
                    "Bond formation and departure overlap as the carbon inverts.")
        elif loss < .999:
            note = "SN1 | Ionization. The leaving group takes the bonding pair. A planar carbocation forms before attack."
        elif self.shift and move < .999:
            kind = "hydride" if self.atoms[self.group]=="H" else "alkyl group"
            note = f"SN1 | A neighboring {kind} moves with its bonding pair. The positive charge moves to the more stabilized carbon."
        else:
            note = "SN1 | Capture. The nucleophile approaches one face of the planar carbocation. Either face can accept its electron pair."
        return labels, np.vstack((pts,incoming)), bonds, partial, near, note
