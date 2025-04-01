import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

class Lattice:
    def __init__(self, Lx, Ly, periodic_x=False, periodic_y=False):
        self.Lx = Lx
        self.Ly = Ly
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.atom_positions = {}
        self.cells = set()
        self.num_sites = None

class LinearLattice(Lattice): 
    def __init__(self, L, periodic=False):
        super().__init__(L, 1, periodic_x=periodic, periodic_y=False)
        self.atom_positions = {i: (i, 0) for i in range(L)}
        self.cells = set([i for i in range(L)])
        self.num_sites = L

    def get_connections(self, locality='nn'): 
        if locality == 'nn':
            nn_pairs = [(i, i + 1) for i in range(self.Lx - 1)]
            if self.periodic_x:
                nn_pairs.append((self.Lx - 1, 0))
            return nn_pairs
        
        elif locality == 'nnn': 
            nnn_pairs = [(i, i + 2) for i in range(self.Lx - 2)]
            if self.periodic_x:
                nnn_pairs.append((self.Lx - 2, 0))
                nnn_pairs.append((self.Lx - 1, 1))
            return nnn_pairs
        
        elif locality == 'a2a': 
            a2a_pairs = [(i,j) for i in range(self.Lx) for j in range(i+1, self.Lx)]
            return a2a_pairs

        else:
            raise ValueError("Unsupported locality type")

class ToricLattice:
    def __init__(self, Lx, Ly):
        """ 
        qubits on horizontal edges are indexed first (left to right, top to bottom),
        followed by vertical edges (left to right, top to bottom).
        Edges connecting boundary vertices are also decorated with qubits due to periodicity.
        (x, y, 'h'/'v') refers to an edge starting from vertex (x, y) and going right or down.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.num_sites = 2 * Lx * Ly

        self.edge_map = {}          # (x, y, 'h'/'v') → index
        self.index_map = {}         # index → (x, y, 'h'/'v')
        self.vertex_terms = []      # list of 4-site lists
        self.plaquette_terms = []   # list of 4-site lists

        self._build_edges()
        self._build_vertex_terms()
        self._build_plaquette_terms()

    def _build_edges(self):
        idx = 0
        for y in range(self.Ly):
            for x in range(self.Lx):
                self.edge_map[(x, y, 'h')] = idx
                self.index_map[idx] = (x, y, 'h')
                idx += 1
        for y in range(self.Ly):
            for x in range(self.Lx):
                self.edge_map[(x, y, 'v')] = idx
                self.index_map[idx] = (x, y, 'v')
                idx += 1

    def _build_vertex_terms(self):
        for y in range(self.Ly):
            for x in range(self.Lx):
                edges = [
                    self.edge_map[(x, y, 'h')],
                    self.edge_map[((x - 1) % self.Lx, y, 'h')],
                    self.edge_map[(x, y, 'v')],
                    self.edge_map[(x, (y - 1) % self.Ly, 'v')],
                ]
                self.vertex_terms.append(edges)

    def _build_plaquette_terms(self):
        for y in range(self.Ly):
            for x in range(self.Lx):
                edges = [
                    self.edge_map[(x, y, 'h')],
                    self.edge_map[((x + 1) % self.Lx, y, 'v')],
                    self.edge_map[(x, (y + 1) % self.Ly, 'h')],
                    self.edge_map[(x, y, 'v')],
                ]
                self.plaquette_terms.append(edges)

    def __getitem__(self, index):
        return self.index_map[index]

    def plot(self):
        raise NotImplementedError("Drawing function not implemented for ToricLattice")
    
class KagomeLattice:
    def __init__(self, row_layout, periodic_x=False, periodic_y=False):
        """
        General Kagome lattice constructed from a list of rows.
        Each row is a dict: {'num_cells': int, 'shift': float}
        Qubits live on A, B, C sites in triangles.
        """
        self.row_layout = row_layout
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y

        self.site_map = OrderedDict()   # (i, j, 'A'/'B'/'C') → index
        self.index_map = {}             # index → (x, y, label)
        self.nn_pairs = set()
        self.triangle_terms = set()
        self.num_sites = 0

        self._place_sites()
        self._build_nn()
        self._build_triangles()

    def _place_sites(self):
        """Places qubit sites at triangle corners, shifted by row_layout."""
        index = 0
        triangle_basis = {
            'A': np.array([0.0, 0.0]),
            'B': np.array([0.5, 0.0]),
            'C': np.array([0.25, 0.5])
        }

        for j, row in enumerate(self.row_layout):
            shift_j = row.get('shift', 0.0)
            for i in range(row['num_cells']):
                for label, offset in triangle_basis.items():
                    x = i + offset[0] + shift_j
                    y = j + offset[1]
                    key = (i, j, label)
                    self.site_map[key] = index
                    self.index_map[index] = (x, y, label)
                    index += 1

        self.num_sites = index

    def _build_nn(self):
        """Build intra- and inter-triangle nearest-neighbor pairs with correct shift logic."""
        triangle_edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]

        # Intra-triangle pairs
        for (i, j, label), idx in self.site_map.items():
            for a, b in triangle_edges:
                if label == a:
                    key_b = (i, j, b)
                    if key_b in self.site_map:
                        idx_b = self.site_map[key_b]
                        self.nn_pairs.add(tuple(sorted((idx, idx_b))))

        # Inter-triangle pairs
        for (i, j, label), idx in self.site_map.items():
            row = self.row_layout[j]
            Lx = row['num_cells']
            shift_j = row.get('shift', 0.0)

            if label == 'B':
                # B(i,j) → A(i+1,j)
                ni = (i + 1) % Lx if self.periodic_x else (i + 1)
                if 0 <= ni < Lx:
                    key = (ni, j, 'A')
                    if key in self.site_map:
                        self.nn_pairs.add(tuple(sorted((idx, self.site_map[key]))))

            elif label == 'C':
                nj = (j + 1) % len(self.row_layout) if self.periodic_y else (j + 1)
                if 0 <= nj < len(self.row_layout):
                    row_next = self.row_layout[nj]
                    shift_next = row_next.get('shift', 0.0)
                    Lx_next = row_next['num_cells']

                    delta = shift_next - shift_j
                    if delta > 0.0: # Row above is shifted right
                        key_A = (i, nj, 'A')
                        key_B = (i - int(delta/0.5), nj, 'B')
                    elif delta < 0.0: # Row above is shifted left
                        key_A = (i + int(-delta/0.5), nj, 'A')
                        key_B = (i, nj, 'B')
                    else:
                        continue

                    for key in [key_A, key_B]:
                        ki, kj, sub = key
                        if self.periodic_x:
                            ki = ki % Lx_next
                        if not (0 <= ki < Lx_next):
                            continue
                        full_key = (ki, kj, sub)
                        if full_key in self.site_map:
                            idx_b = self.site_map[full_key]
                            self.nn_pairs.add(tuple(sorted((idx, idx_b))))

    def _build_triangles(self):
        """Add in-cell and inter-row triangles with consistent orientation."""
        # Intra-cell ABC triangles
        for (i, j, _) in self.site_map:
            try:
                a = self.site_map[(i, j, 'A')]
                b = self.site_map[(i, j, 'B')]
                c = self.site_map[(i, j, 'C')]
                self.triangle_terms.add((a, b, c))  # counter-clockwise

            except KeyError:
                continue

        # Inter-row triangles formed between C(i,j) and row j+1
        for (i, j, label) in self.site_map:
            if label != 'C':
                continue

            shift_j = self.row_layout[j].get('shift', 0.0)
            nj = (j + 1) % len(self.row_layout) if self.periodic_y else j + 1
            if not (0 <= nj < len(self.row_layout)):
                continue

            shift_next = self.row_layout[nj].get('shift', 0.0)
            delta = shift_next - shift_j
            Lx_next = self.row_layout[nj]['num_cells']

            # Define triangle based on relative shift
            if delta > 0.0: 
                # Triangle: C(i,j), A(i, j+1), B(i-1, j+1)
                key_A = (i, nj, 'A')
                key_B = (i - int(delta/0.5), nj, 'B')
            elif delta < 0.0:
                # Triangle: C(i,j), A(i+1, j+1), B(i, j+1)
                key_A = (i + int(-delta/0.5), nj, 'A')
                key_B = (i, nj, 'B')
            else:
                continue

            # Index wrap + bounds
            triangle = []
            for key in [key_A, key_B]:
                ki, kj, sub = key
                if self.periodic_x:
                    ki = ki % Lx_next
                if not (0 <= ki < Lx_next):
                    break
                full_key = (ki, kj, sub)
                if full_key in self.site_map:
                    triangle.append(self.site_map[full_key])
            # Add C(i,j)
            triangle.insert(0, self.site_map[(i, j, 'C')])

            if len(triangle) == 3:
                self.triangle_terms.add(tuple(triangle))  # keep orientation consistent

    def __getitem__(self, index):
        return self.index_map[index]

    def get_connections(self, locality='nn'):
        if locality == 'nn':
            return list(self.nn_pairs)
        elif locality == 'triangle':
            return self.triangle_terms
        else:
            raise ValueError(f"Unsupported locality: {locality}")

    def plot(self, show_labels=True, show_triangles=True, figsize=(8, 6), save_path=None):
        _, ax = plt.subplots(figsize=figsize)
        colors = {'A': 'blue', 'B': 'green', 'C': 'red'}

        for idx, (x, y, label) in self.index_map.items():
            ax.scatter(x, y, s=100, color=colors[label], edgecolor='black', zorder=2)
            if show_labels:
                ax.text(x, y, str(idx), ha='center', va='center', color='white', fontsize=8)

        for i, j in self.nn_pairs:
            x1, y1, _ = self.index_map[i]
            x2, y2, _ = self.index_map[j]
            ax.plot([x1, x2], [y1, y2], 'k-', lw=1, zorder=1)

        if show_triangles:
            for a, b, c in self.triangle_terms:
                xa, ya, _ = self.index_map[a]
                xb, yb, _ = self.index_map[b]
                xc, yc, _ = self.index_map[c]
                ax.plot([xa, xb, xc, xa], [ya, yb, yc, ya], 'gray', lw=0.5, linestyle='--', zorder=0)

        ax.set_aspect('equal')
        ax.axis('off')
        plt.title("Kagome Lattice")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

    def spruce(self, deleted_qubits):
        deleted_set = set(deleted_qubits)

        # Step 1: Remove any nn_pair or triangle term that includes a deleted qubit
        self.nn_pairs = {pair for pair in self.nn_pairs if not deleted_set.intersection(pair)}
        self.triangle_terms = {tri for tri in self.triangle_terms if not deleted_set.intersection(tri)}

        # Step 2: Create a new index mapping for the remaining qubits
        remaining_indices = sorted(set(self.index_map.keys()) - deleted_set)
        index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_indices)}

        # Step 3: Update site_map and index_map
        new_site_map = OrderedDict()
        new_index_map = {}

        for key, old_idx in self.site_map.items():
            if old_idx in index_remap:
                new_idx = index_remap[old_idx]
                new_site_map[key] = new_idx
                new_index_map[new_idx] = self.index_map[old_idx]

        self.site_map = new_site_map
        self.index_map = new_index_map
        self.num_sites = len(self.index_map)

        # Step 4: Remap nn_pairs and triangle_terms with new indices
        self.nn_pairs = {tuple(sorted((index_remap[i], index_remap[j]))) for i, j in self.nn_pairs}
        self.triangle_terms = {tuple(index_remap[i] for i in tri) for tri in self.triangle_terms}

    def __str__(self):
        return f"KagomeLattice with {self.num_sites} sites, {len(self.triangle_terms)} triangles"