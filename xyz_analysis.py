import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Common bond lengths in Angstroms
BOND_LENGTHS = {
    ('H', 'H'): 0.74,
    ('C', 'H'): 1.09,
    ('H', 'N'): 1.01,
    ('H', 'O'): 0.96,
    ('C', 'C'): 1.54,
    ('C', 'N'): 1.47,
    ('C', 'O'): 1.40,
    ('N', 'N'): 1.45,
    ('N', 'O'): 1.40,
    ('O', 'O'): 1.48,
}

# Common hydrogen bond parameters
# Format: (donor_element, H, acceptor_element): (max_distance, min_angle)
# Distance in Angstroms, angles in degrees
H_BOND_PARAMS = {
    # O-H...O hydrogen bonds
    ('O', 'H', 'O'): (3.5, 120),
    # N-H...O hydrogen bonds
    ('N', 'H', 'O'): (3.5, 120),
    # O-H...N hydrogen bonds
    ('O', 'H', 'N'): (3.5, 120),
    # N-H...N hydrogen bonds
    ('N', 'H', 'N'): (3.5, 120),
    # C-H...O hydrogen bonds (weaker)
    ('C', 'H', 'O'): (3.2, 90),
    # O-H...F hydrogen bonds
    ('O', 'H', 'F'): (3.2, 120),
    # N-H...F hydrogen bonds
    ('N', 'H', 'F'): (3.2, 120),
    # O-H...S hydrogen bonds
    ('O', 'H', 'S'): (3.8, 110),
    # N-H...S hydrogen bonds
    ('N', 'H', 'S'): (3.8, 110),
}

# ============= UTILITY FUNCTIONS =============

def cal_dis(coords, boxsize):
    """
    Calculate distance matrix between atoms with periodic boundary conditions.
    
    Args:
        coords: Atom coordinates with shape (Natoms, 3)
        boxsize: Simulation box dimensions with shape (3,)
        
    Returns:
        Distance matrix with shape (Natoms, Natoms)
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    diff = diff - boxsize * np.round(diff / boxsize)
    dist = np.linalg.norm(diff, axis=-1)
    return dist

def find_bonds(coords, elements, boxsize, ratio=1.2, bond_lengths=BOND_LENGTHS):
    """
    Find bonds between atoms based on distance criteria.
    
    Args:
        coords: Atom coordinates with shape (Natoms, 3)
        elements: List of element types for each atom
        boxsize: Simulation box dimensions with shape (3,)
        ratio: Bond length multiplier to determine bonding cutoff
        bond_lengths: Dictionary of reference bond lengths
        
    Returns:
        List of tuples (i, j) representing bonds between atoms
    """
    num_atoms = coords.shape[0]
    bonds = []
    dist_matrix = cal_dis(coords, boxsize)
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom1, atom2 = elements[i], elements[j]
            bond_key = tuple(sorted((atom1, atom2)))
            if bond_key in bond_lengths:
                bond_length = bond_lengths[bond_key] * ratio
                if dist_matrix[i, j] <= bond_length:
                    bonds.append((i, j))
                    
    return bonds

def bonds_to_bond_dict(bonds):
    """
    Convert a list of bonds to a bond dictionary.
    
    Args:
        bonds: List of tuples (i, j) representing bonds between atoms
        
    Returns:
        Dictionary mapping atom indices to their bonded neighbors
    """
    bond_dict = {}
    for i, j in bonds:
        if i not in bond_dict:
            bond_dict[i] = []
        if j not in bond_dict:
            bond_dict[j] = []
        bond_dict[i].append(j)
        bond_dict[j].append(i)
    return bond_dict

def read_xyz(filename):
    """
    Read multiple frames from an XYZ file.
    
    Args:
        filename: Path to XYZ file
        
    Returns:
        tuple: (frames, elements) where frames is a numpy array of coordinates
               and elements is a list of element types for each frame
    """
    frames, all_elements = [], []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        i += 1  # Skip comment line
        i += 1
        
        atom_coords = []
        elements = []
        for _ in range(num_atoms):
            parts = lines[i].split()
            elements.append(parts[0])
            atom_coords.append(list(map(float, parts[1:])))
            i += 1

        frames.append(np.array(atom_coords))
        all_elements.append(elements)

    return np.array(frames), all_elements

# ============= COVALENT BOND ANALYSIS =============

def _process_frame(frame_data, elements_list, ref_bonds, box_size, ratio, bond_lengths, reference_frame):
    """
    Process a single frame to find bonds and calculate changes.
    This function must be at module level to work with multiprocessing.
    """
    frame_idx, coords = frame_data
    elements = elements_list[frame_idx]
    
    # Find bonds in current frame
    current_bonds = set(find_bonds(coords, elements, box_size, ratio, bond_lengths))
    
    # Compute bond changes relative to reference frame
    if frame_idx == reference_frame:
        change_data = {
            "frame": frame_idx,
            "added_bonds": [],
            "broken_bonds": [],
        }
    else:
        added_bonds = current_bonds - ref_bonds
        broken_bonds = ref_bonds - current_bonds
        change_data = {
            "frame": frame_idx,
            "added_bonds": list(added_bonds),
            "broken_bonds": list(broken_bonds),
        }
    
    return frame_idx, current_bonds, change_data

class MolecularBondAnalyzer:
    """
    Class for analyzing molecular bond changes throughout a trajectory.
    """
    
    def __init__(self, bond_lengths=BOND_LENGTHS):
        """
        Initialize the analyzer with bond length parameters.
        
        Args:
            bond_lengths: Dictionary of reference bond lengths
        """
        self.bond_lengths = bond_lengths
        self.trajectory = None
        self.elements_list = None
        self.changes = None
        self.bond_history = None
    
    def load_trajectory(self, filename):
        """
        Load trajectory from XYZ file.
        
        Args:
            filename: Path to XYZ file
            
        Returns:
            self: For method chaining
        """
        self.trajectory, self.elements_list = read_xyz(filename)
        return self
    
    def set_trajectory(self, trajectory, elements_list):
        """
        Set trajectory data directly.
        
        Args:
            trajectory: Numpy array of coordinates with shape (frames, atoms, 3)
            elements_list: List of element types for each frame
            
        Returns:
            self: For method chaining
        """
        self.trajectory = trajectory
        self.elements_list = elements_list
        return self
    
    def analyze_bond_changes(self, box_size, ratio=1.2, reference_frame=0, show_progress=True):
        """
        Analyze bond changes throughout the trajectory.
        
        Args:
            box_size: Simulation box dimensions with shape (3,)
            ratio: Bond length multiplier to determine bonding cutoff
            reference_frame: Frame to use as reference for bond changes
            show_progress: Whether to show progress bar
            
        Returns:
            tuple: (changes, bond_history) where changes is a list of dictionaries
                  and bond_history is a list of sets of bonds for each frame
        """
        if not hasattr(self, 'trajectory') or self.trajectory is None:
            raise ValueError("No trajectory data. Call load_trajectory() or set_trajectory() first.")
            
        bond_history = []
        changes = []
        
        # Iterator with or without progress bar
        iterator = enumerate(self.trajectory)
        if show_progress:
            iterator = tqdm(iterator, total=self.trajectory.shape[0], desc="Analyzing bonds")
            
        # Process each frame
        for frame_idx, coords in iterator:
            elements = self.elements_list[frame_idx]
            
            # Find bonds in current frame
            current_bonds = set(find_bonds(coords, elements, box_size, 
                                         ratio, self.bond_lengths))
            bond_history.append(current_bonds)
            
            # Skip computing changes for the reference frame
            if frame_idx == reference_frame:
                changes.append({
                    "frame": frame_idx,
                    "added_bonds": [],
                    "broken_bonds": [],
                })
                continue
                
            # Compute bond changes relative to reference frame
            ref_bonds = bond_history[reference_frame]
            added_bonds = current_bonds - ref_bonds
            broken_bonds = ref_bonds - current_bonds
            
            changes.append({
                "frame": frame_idx,
                "added_bonds": list(added_bonds),
                "broken_bonds": list(broken_bonds),
            })
            
        self.changes = changes
        self.bond_history = bond_history
        
        return changes, bond_history
    
    def analyze_bond_changes_parallel(self, box_size, ratio=1.2, reference_frame=0, show_progress=True, n_processes=None):
        """
        Analyze bond changes throughout the trajectory using parallel processing.
        
        Args:
            box_size: Simulation box dimensions with shape (3,)
            ratio: Bond length multiplier to determine bonding cutoff
            reference_frame: Frame to use as reference for bond changes
            show_progress: Whether to show progress bar
            n_processes: Number of processes to use (defaults to CPU count)
            
        Returns:
            tuple: (changes, bond_history) where changes is a list of dictionaries
                  and bond_history is a list of sets of bonds for each frame
        """
        if not hasattr(self, 'trajectory') or self.trajectory is None:
            raise ValueError("No trajectory data. Call load_trajectory() or set_trajectory() first.")
        
        # First, process the reference frame to get reference bonds
        ref_elements = self.elements_list[reference_frame]
        ref_coords = self.trajectory[reference_frame]
        ref_bonds = set(find_bonds(ref_coords, ref_elements, box_size, ratio, self.bond_lengths))
        
        # Create a pool of workers
        n_processes = n_processes or mp.cpu_count()
        
        # Prepare the frame data
        frame_data = list(enumerate(self.trajectory))
        
        # Create a partial function with fixed arguments
        worker_func = partial(
            _process_frame,
            elements_list=self.elements_list,
            ref_bonds=ref_bonds,
            box_size=box_size,
            ratio=ratio,
            bond_lengths=self.bond_lengths,
            reference_frame=reference_frame
        )
        
        # Process frames in parallel
        with mp.Pool(processes=n_processes) as pool:
            if show_progress:
                results = list(tqdm(pool.imap(worker_func, frame_data), total=len(frame_data), desc="Analyzing bonds"))
            else:
                results = pool.map(worker_func, frame_data)
        
        # Sort results by frame index and extract bond data
        results.sort(key=lambda x: x[0])  # Sort by frame index
        
        # Extract sorted data
        bond_history = [bonds for _, bonds, _ in results]
        changes = [change for _, _, change in results]
        
        # Store results as instance attributes
        self.changes = changes
        self.bond_history = bond_history
        
        return changes, bond_history

    def get_bond_evolution(self, bond):
        """
        Track the presence of a specific bond throughout the trajectory.
        
        Args:
            bond: Tuple (i, j) representing the bond to track
            
        Returns:
            List of boolean values indicating bond presence in each frame
        """
        if not hasattr(self, 'bond_history') or self.bond_history is None:
            raise ValueError("No bond history. Call analyze_bond_changes() first.")
            
        bond = tuple(sorted(bond))  # Ensure consistent order
        return [bond in bonds for bonds in self.bond_history]
    
    def summarize_changes(self):
        """
        Summarize bond changes throughout the trajectory.
        
        Returns:
            dict: Summary statistics about bond changes
        """
        if not hasattr(self, 'changes') or self.changes is None:
            raise ValueError("No change data. Call analyze_bond_changes() first.")
            
        total_frames = len(self.changes)
        frames_with_changes = sum(1 for c in self.changes if c["added_bonds"] or c["broken_bonds"])
        
        # Count total unique bonds formed and broken
        all_added = set()
        all_broken = set()
        for c in self.changes:
            all_added.update(c["added_bonds"])
            all_broken.update(c["broken_bonds"])
            
        return {
            "total_frames": total_frames,
            "frames_with_changes": frames_with_changes,
            "percent_frames_with_changes": (frames_with_changes / total_frames) * 100,
            "unique_bonds_formed": len(all_added),
            "unique_bonds_broken": len(all_broken)
        }
    
    def get_bond_dict_for_frame(self, frame_idx=0):
        """
        Get a bond dictionary for a specific frame.
        
        Args:
            frame_idx: Index of the frame to use
            
        Returns:
            Dictionary mapping atom indices to their bonded neighbors
        """
        if not hasattr(self, 'bond_history') or self.bond_history is None:
            raise ValueError("No bond history. Call analyze_bond_changes() first.")
        
        if frame_idx >= len(self.bond_history):
            raise ValueError(f"Frame index {frame_idx} out of range.")
        
        bonds = list(self.bond_history[frame_idx])
        return bonds_to_bond_dict(bonds)

# ============= HYDROGEN BOND ANALYSIS =============

def _process_hbonds_frame(frame_data, elements_list, box_size, h_bond_params, bond_dict, donor_indices, distance_cutoff):
    """
    Process a single frame to find hydrogen bonds.
    
    Args:
        frame_data: Tuple of (frame_idx, coords)
        elements_list: List of element types for each atom
        box_size: Box dimensions for periodic boundary conditions
        h_bond_params: Dictionary of hydrogen bond parameters
        bond_dict: Dictionary mapping atom indices to bonded neighbors
        donor_indices: List of indices for potential donor atoms
        distance_cutoff: Maximum cutoff distance for hydrogen bond screening
    
    Returns:
        Tuple of (frame_idx, list of hydrogen bonds)
    """
    frame_idx, coords = frame_data
    elements = elements_list[frame_idx]
    h_bonds = []
    
    # Calculate all pairwise distances with PBC
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    diff = diff - box_size * np.round(diff / box_size)
    distances = np.linalg.norm(diff, axis=-1)
    
    # For each potential donor atom
    for donor_idx in donor_indices:
        donor_element = elements[donor_idx]
        
        # Find bonded hydrogens
        for h_idx in bond_dict.get(donor_idx, []):
            if elements[h_idx] != 'H':
                continue
                
            # Vector from donor to hydrogen
            d_h_vec = coords[h_idx] - coords[donor_idx]
            d_h_vec = d_h_vec - box_size * np.round(d_h_vec / box_size)  # Apply PBC
            d_h_dist = np.linalg.norm(d_h_vec)
            
            if d_h_dist == 0:  # Avoid division by zero
                continue
                
            # Normalize the D-H vector
            d_h_vec = d_h_vec / d_h_dist
            
            # Find potential acceptors
            for acceptor_idx in range(len(elements)):
                # Skip self, donor, and atoms bonded to donor
                if acceptor_idx == donor_idx or acceptor_idx == h_idx or acceptor_idx in bond_dict.get(donor_idx, []):
                    continue
                    
                acceptor_element = elements[acceptor_idx]
                
                # Check if this is a valid hydrogen bond type
                h_bond_key = (donor_element, 'H', acceptor_element)
                if h_bond_key not in h_bond_params:
                    continue
                    
                max_distance, min_angle_deg = h_bond_params[h_bond_key]
                
                # Distance between hydrogen and acceptor
                h_a_dist = distances[h_idx, acceptor_idx]
                
                # Quick distance check before doing angle calculations
                if h_a_dist > max_distance:
                    continue
                    
                # Vector from hydrogen to acceptor
                h_a_vec = coords[acceptor_idx] - coords[h_idx]
                h_a_vec = h_a_vec - box_size * np.round(h_a_vec / box_size)  # Apply PBC
                
                if np.linalg.norm(h_a_vec) == 0:  # Avoid division by zero
                    continue
                    
                # Normalize the H-A vector
                h_a_vec = h_a_vec / np.linalg.norm(h_a_vec)
                
                # Calculate the D-H...A angle in degrees
                cos_angle = np.dot(d_h_vec, h_a_vec)
                angle_deg = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                
                # Check if this is a hydrogen bond based on distance and angle
                if h_a_dist <= max_distance and angle_deg >= min_angle_deg:
                    h_bonds.append({
                        'donor_idx': donor_idx,
                        'donor_element': donor_element,
                        'hydrogen_idx': h_idx,
                        'acceptor_idx': acceptor_idx,
                        'acceptor_element': acceptor_element,
                        'distance': h_a_dist,
                        'angle': angle_deg
                    })
    
    return frame_idx, h_bonds

class HydrogenBondAnalyzer:
    """
    Class for analyzing hydrogen bonds in molecular trajectories.
    Can be used with MolecularBondAnalyzer or standalone.
    """
    
    def __init__(self, trajectory=None, elements_list=None, h_bond_params=H_BOND_PARAMS):
        """
        Initialize the hydrogen bond analyzer.
        
        Args:
            trajectory: Optional numpy array of coordinates with shape (frames, atoms, 3)
            elements_list: Optional list of element types for each frame
            h_bond_params: Dictionary of hydrogen bond parameters
        """
        self.trajectory = trajectory
        self.elements_list = elements_list
        self.h_bond_params = h_bond_params
        self.h_bonds_history = None
    
    def load_trajectory(self, filename):
        """
        Load trajectory from XYZ file.
        
        Args:
            filename: Path to XYZ file
            
        Returns:
            self: For method chaining
        """
        self.trajectory, self.elements_list = read_xyz(filename)
        return self
    
    def set_trajectory(self, trajectory, elements_list):
        """
        Set trajectory data.
        
        Args:
            trajectory: Numpy array of coordinates with shape (frames, atoms, 3)
            elements_list: List of element types for each frame
            
        Returns:
            self: For method chaining
        """
        self.trajectory = trajectory
        self.elements_list = elements_list
        return self
    
    def find_hydrogen_bonds(self, box_size, bond_dict=None, covalent_bonds=None, 
                           bond_analyzer=None, frame_idx=0,
                           donor_indices=None, focus_molecule_indices=None,
                           distance_cutoff=4.0, show_progress=True, n_processes=None):
        """
        Find hydrogen bonds throughout the trajectory.
        
        Args:
            box_size: Simulation box dimensions with shape (3,)
            bond_dict: Optional dictionary mapping atom indices to their bonded neighbors
            covalent_bonds: Optional list of (i,j) tuples representing covalent bonds
            bond_analyzer: Optional MolecularBondAnalyzer instance to get bond data from
            frame_idx: Frame index to use when getting bond data from bond_analyzer
            donor_indices: Optional list of indices for potential donor atoms
            focus_molecule_indices: Optional list of atom indices to focus analysis on
            distance_cutoff: Maximum cutoff distance for hydrogen bond screening
            show_progress: Whether to show progress bar
            n_processes: Number of processes to use for parallel processing
                       
        Returns:
            h_bonds_history: List of lists of hydrogen bonds for each frame
        """
        if self.trajectory is None or self.elements_list is None:
            raise ValueError("No trajectory data. Call set_trajectory() or load_trajectory() first.")
        
        # Create bond dictionary using the provided source
        if bond_dict is None:
            if bond_analyzer is not None:
                bond_dict = bond_analyzer.get_bond_dict_for_frame(frame_idx)
            elif covalent_bonds is not None:
                bond_dict = bonds_to_bond_dict(covalent_bonds)
            else:
                raise ValueError("Either bond_dict, covalent_bonds, or bond_analyzer must be provided.")
        
        # Determine donor atoms if not provided
        if donor_indices is None:
            # Get elements from first frame as representative
            first_frame_elements = self.elements_list[0]
            
            # Find all potential donor elements (N, O, etc.) from the h_bond_params
            donor_elements = set(donor_type for donor_type, _, _ in self.h_bond_params.keys())
            
            # Create list of all potential donor atom indices
            donor_indices = [i for i, element in enumerate(first_frame_elements) 
                           if element in donor_elements]
            
            # Filter by focus molecule if provided
            if focus_molecule_indices is not None:
                donor_indices = [i for i in donor_indices if i in focus_molecule_indices]
        
        # Prepare for parallel processing
        frame_data = list(enumerate(self.trajectory))
        worker_func = partial(
            _process_hbonds_frame,
            elements_list=self.elements_list,
            box_size=box_size,
            h_bond_params=self.h_bond_params,
            bond_dict=bond_dict,
            donor_indices=donor_indices,
            distance_cutoff=distance_cutoff
        )
        
        # Process frames
        if n_processes and n_processes > 1:
            # Parallel processing
            with mp.Pool(processes=n_processes) as pool:
                if show_progress:
                    results = list(tqdm(pool.imap(worker_func, frame_data), 
                                       total=len(frame_data), 
                                       desc="Analyzing H-bonds"))
                else:
                    results = pool.map(worker_func, frame_data)
        else:
            # Sequential processing
            results = []
            iterator = frame_data
            if show_progress:
                iterator = tqdm(iterator, total=len(frame_data), desc="Analyzing H-bonds")
            
            for frame_data_item in iterator:
                results.append(worker_func(frame_data_item))
        
        # Sort results by frame index
        results.sort(key=lambda x: x[0])
        
        # Extract hydrogen bonds data
        h_bonds_history = [h_bonds for _, h_bonds in results]
        self.h_bonds_history = h_bonds_history
        
        return h_bonds_history
    
    def analyze_hydrogen_bonds(self, target_indices=None, min_persistence=0.5):
        """
        Analyze hydrogen bonds statistics.
        
        Args:
            target_indices: Optional list of atom indices to focus on
            min_persistence: Minimum fraction of frames a hydrogen bond must appear in to be considered persistent
            
        Returns:
            dict: Dictionary of hydrogen bond statistics
        """
        if self.h_bonds_history is None:
            raise ValueError("No hydrogen bond data. Call find_hydrogen_bonds() first.")
        
        total_frames = len(self.h_bonds_history)
        
        # Count hydrogen bonds per frame
        h_bonds_per_frame = [len(frame_bonds) for frame_bonds in self.h_bonds_history]
        
        # Track unique hydrogen bonds and count occurrences
        h_bond_occurrences = {}
        for frame_idx, frame_bonds in enumerate(self.h_bonds_history):
            for h_bond in frame_bonds:
                # Filter by target indices if provided
                if target_indices is not None:
                    if (h_bond['donor_idx'] not in target_indices and 
                        h_bond['hydrogen_idx'] not in target_indices and 
                        h_bond['acceptor_idx'] not in target_indices):
                        continue
                
                # Use a tuple of (donor, hydrogen, acceptor) as the key
                key = (h_bond['donor_idx'], h_bond['hydrogen_idx'], h_bond['acceptor_idx'])
                
                if key not in h_bond_occurrences:
                    h_bond_occurrences[key] = {
                        'count': 0,
                        'frames': [],
                        'distances': [],
                        'angles': [],
                        'donor_element': h_bond['donor_element'],
                        'acceptor_element': h_bond['acceptor_element']
                    }
                
                h_bond_occurrences[key]['count'] += 1
                h_bond_occurrences[key]['frames'].append(frame_idx)
                h_bond_occurrences[key]['distances'].append(h_bond['distance'])
                h_bond_occurrences[key]['angles'].append(h_bond['angle'])
        
        # Calculate persistence and statistics for each hydrogen bond
        persistent_h_bonds = {}
        for key, data in h_bond_occurrences.items():
            persistence = data['count'] / total_frames
            
            if persistence >= min_persistence:
                persistent_h_bonds[key] = {
                    'donor_idx': key[0],
                    'hydrogen_idx': key[1],
                    'acceptor_idx': key[2],
                    'donor_element': data['donor_element'],
                    'acceptor_element': data['acceptor_element'],
                    'persistence': persistence,
                    'avg_distance': np.mean(data['distances']),
                    'std_distance': np.std(data['distances']),
                    'avg_angle': np.mean(data['angles']),
                    'std_angle': np.std(data['angles']),
                    'frames': data['frames']
                }
        
        # Prepare summary statistics
        summary = {
            'total_frames': total_frames,
            'avg_h_bonds_per_frame': np.mean(h_bonds_per_frame),
            'max_h_bonds_per_frame': np.max(h_bonds_per_frame),
            'min_h_bonds_per_frame': np.min(h_bonds_per_frame),
            'std_h_bonds_per_frame': np.std(h_bonds_per_frame),
            'unique_h_bonds': len(h_bond_occurrences),
            'persistent_h_bonds': len(persistent_h_bonds),
            'persistent_h_bonds_data': persistent_h_bonds
        }
        
        return summary
    
    def get_hydrogen_bonds_involving_atom(self, atom_idx):
        """
        Get all hydrogen bonds involving a specific atom.
        
        Args:
            atom_idx: Index of the atom of interest
            
        Returns:
            list: List of hydrogen bonds involving the specified atom across all frames
        """
        if self.h_bonds_history is None:
            raise ValueError("No hydrogen bond data. Call find_hydrogen_bonds() first.")
        
        h_bonds_involving_atom = []
        
        for frame_idx, frame_bonds in enumerate(self.h_bonds_history):
            for h_bond in frame_bonds:
                if (h_bond['donor_idx'] == atom_idx or 
                    h_bond['hydrogen_idx'] == atom_idx or 
                    h_bond['acceptor_idx'] == atom_idx):
                    
                    # Add frame index to the hydrogen bond info
                    h_bond_with_frame = h_bond.copy()
                    h_bond_with_frame['frame'] = frame_idx
                    h_bonds_involving_atom.append(h_bond_with_frame)
        
        return h_bonds_involving_atom

# ============= INTEGRATED ANALYZER =============

class MolecularAnalyzer:
    """
    Integrated class for analyzing both covalent bonds and hydrogen bonds.
    """
    
    def __init__(self, bond_lengths=BOND_LENGTHS, h_bond_params=H_BOND_PARAMS):
        """
        Initialize the integrated analyzer.
        
        Args:
            bond_lengths: Dictionary of reference bond lengths
            h_bond_params: Dictionary of hydrogen bond parameters
        """
        self.bond_analyzer = MolecularBondAnalyzer(bond_lengths)
        self.h_bond_analyzer = HydrogenBondAnalyzer(h_bond_params=h_bond_params)
        self.trajectory = None
        self.elements_list = None
        
    def load_trajectory(self, filename):
        """
        Load trajectory from XYZ file.
        
        Args:
            filename: Path to XYZ file
            
        Returns:
            self: For method chaining
        """
        self.trajectory, self.elements_list = read_xyz(filename)
        self.bond_analyzer.set_trajectory(self.trajectory, self.elements_list)
        self.h_bond_analyzer.set_trajectory(self.trajectory, self.elements_list)
        return self
    
    def set_trajectory(self, trajectory, elements_list):
        """
        Set trajectory data directly.
        
        Args:
            trajectory: Numpy array of coordinates with shape (frames, atoms, 3)
            elements_list: List of element types for each frame
            
        Returns:
            self: For method chaining
        """
        self.trajectory = trajectory
        self.elements_list = elements_list
        self.bond_analyzer.set_trajectory(trajectory, elements_list)
        self.h_bond_analyzer.set_trajectory(trajectory, elements_list)
        return self
        
    def analyze_trajectory(self, box_size, covalent_bond_ratio=1.2, 
                          reference_frame=0, h_bond_distance_cutoff=4.0,
                          min_h_bond_persistence=0.5, target_indices=None,
                          n_processes=None, show_progress=True):
        """
        Perform comprehensive analysis of both covalent and hydrogen bonds.
        
        Args:
            box_size: Simulation box dimensions with shape (3,)
            covalent_bond_ratio: Bond length multiplier for covalent bonds
            reference_frame: Frame to use as reference for bond changes
            h_bond_distance_cutoff: Maximum cutoff distance for hydrogen bond screening
            min_h_bond_persistence: Minimum fraction of frames for persistent hydrogen bonds
            target_indices: Optional list of atom indices to focus analysis on
            n_processes: Number of processes for parallel computation
            show_progress: Whether to show progress bars
            
        Returns:
            dict: Dictionary containing analysis results
        """
        if self.trajectory is None or self.elements_list is None:
            raise ValueError("No trajectory data. Call load_trajectory() or set_trajectory() first.")
            
        # Analyze covalent bonds
        if n_processes and n_processes > 1:
            bond_changes, bond_history = self.bond_analyzer.analyze_bond_changes_parallel(
                box_size, ratio=covalent_bond_ratio, reference_frame=reference_frame,
                show_progress=show_progress, n_processes=n_processes
            )
        else:
            bond_changes, bond_history = self.bond_analyzer.analyze_bond_changes(
                box_size, ratio=covalent_bond_ratio, reference_frame=reference_frame,
                show_progress=show_progress
            )
            
        # Get bond dictionary for hydrogen bond analysis
        bond_dict = self.bond_analyzer.get_bond_dict_for_frame(reference_frame)
        
        # Analyze hydrogen bonds
        h_bonds_history = self.h_bond_analyzer.find_hydrogen_bonds(
            box_size, bond_dict=bond_dict, frame_idx=reference_frame,
            focus_molecule_indices=target_indices, distance_cutoff=h_bond_distance_cutoff,
            show_progress=show_progress, n_processes=n_processes
        )
        
        # Analyze hydrogen bond statistics
        h_bond_stats = self.h_bond_analyzer.analyze_hydrogen_bonds(
            target_indices=target_indices, min_persistence=min_h_bond_persistence
        )
        
        # Get summary of covalent bond changes
        bond_summary = self.bond_analyzer.summarize_changes()
        
        # Combine all results
        return {
            "bond_changes": bond_changes,
            "bond_history": bond_history,
            "bond_summary": bond_summary,
            "h_bonds_history": h_bonds_history,
            "h_bond_stats": h_bond_stats
        }
    
    def identify_reactive_sites(self, box_size, covalent_bond_ratio=1.2, 
                              min_change_frequency=0.1, show_progress=True):
        """
        Identify reactive sites in the molecule based on bond formation/breaking frequency.
        
        Args:
            box_size: Simulation box dimensions with shape (3,)
            covalent_bond_ratio: Bond length multiplier for covalent bonds
            min_change_frequency: Minimum frequency of changes to consider a site reactive
            show_progress: Whether to show progress bar
            
        Returns:
            dict: Dictionary mapping atom indices to their reactivity metrics
        """
        if self.trajectory is None or self.elements_list is None:
            raise ValueError("No trajectory data. Call load_trajectory() or set_trajectory() first.")
            
        # Ensure bond analysis has been performed
        if not hasattr(self.bond_analyzer, 'changes') or self.bond_analyzer.changes is None:
            self.bond_analyzer.analyze_bond_changes(
                box_size, ratio=covalent_bond_ratio, show_progress=show_progress
            )
            
        # Count how many times each atom is involved in bond changes
        atom_change_counts = {}
        total_frames = len(self.bond_analyzer.changes)
        
        # Process bond changes
        for change_data in self.bond_analyzer.changes:
            # Process added bonds
            for bond in change_data["added_bonds"]:
                atom1, atom2 = bond
                
                if atom1 not in atom_change_counts:
                    atom_change_counts[atom1] = {"added": 0, "broken": 0}
                if atom2 not in atom_change_counts:
                    atom_change_counts[atom2] = {"added": 0, "broken": 0}
                    
                atom_change_counts[atom1]["added"] += 1
                atom_change_counts[atom2]["added"] += 1
                
            # Process broken bonds
            for bond in change_data["broken_bonds"]:
                atom1, atom2 = bond
                
                if atom1 not in atom_change_counts:
                    atom_change_counts[atom1] = {"added": 0, "broken": 0}
                if atom2 not in atom_change_counts:
                    atom_change_counts[atom2] = {"added": 0, "broken": 0}
                    
                atom_change_counts[atom1]["broken"] += 1
                atom_change_counts[atom2]["broken"] += 1
        
        # Calculate reactivity metrics
        reactive_sites = {}
        for atom_idx, counts in atom_change_counts.items():
            total_changes = counts["added"] + counts["broken"]
            frequency = total_changes / total_frames
            
            if frequency >= min_change_frequency:
                element = self.elements_list[0][atom_idx]  # Get element from first frame
                reactive_sites[atom_idx] = {
                    "element": element,
                    "change_frequency": frequency,
                    "bond_formations": counts["added"],
                    "bond_breakings": counts["broken"],
                    "total_changes": total_changes
                }
        
        return reactive_sites
    
    def analyze_molecular_interactions(self, box_size, focus_indices, 
                                     covalent_bond_ratio=1.2, h_bond_distance_cutoff=4.0,
                                     n_processes=None, show_progress=True):
        """
        Analyze interactions between a specific molecule/group and its environment.
        
        Args:
            box_size: Simulation box dimensions with shape (3,)
            focus_indices: List of atom indices to focus on
            covalent_bond_ratio: Bond length multiplier for covalent bonds
            h_bond_distance_cutoff: Maximum cutoff distance for hydrogen bond screening
            n_processes: Number of processes for parallel computation
            show_progress: Whether to show progress bars
            
        Returns:
            dict: Dictionary containing interaction analysis results
        """
        if self.trajectory is None or self.elements_list is None:
            raise ValueError("No trajectory data. Call load_trajectory() or set_trajectory() first.")
            
        # Ensure bond analysis has been performed
        if not hasattr(self.bond_analyzer, 'bond_history') or self.bond_analyzer.bond_history is None:
            if n_processes and n_processes > 1:
                self.bond_analyzer.analyze_bond_changes_parallel(
                    box_size, ratio=covalent_bond_ratio, show_progress=show_progress,
                    n_processes=n_processes
                )
            else:
                self.bond_analyzer.analyze_bond_changes(
                    box_size, ratio=covalent_bond_ratio, show_progress=show_progress
                )
        
        # Find inter-molecular covalent bonds (bonds between focus group and environment)
        inter_molecular_bonds = []
        
        for frame_idx, bonds in enumerate(self.bond_analyzer.bond_history):
            frame_inter_bonds = []
            
            for bond in bonds:
                atom1, atom2 = bond
                
                # Check if bond connects focus group with environment
                if (atom1 in focus_indices and atom2 not in focus_indices) or \
                   (atom2 in focus_indices and atom1 not in focus_indices):
                    elements = self.elements_list[frame_idx]
                    frame_inter_bonds.append({
                        'frame': frame_idx,
                        'bond': (atom1, atom2),
                        'elements': (elements[atom1], elements[atom2])
                    })
            
            inter_molecular_bonds.append(frame_inter_bonds)
        
        # Get bond dictionary for first frame
        bond_dict = self.bond_analyzer.get_bond_dict_for_frame(0)
        
        # Find hydrogen bonds involving focus group
        if not hasattr(self.h_bond_analyzer, 'h_bonds_history') or self.h_bond_analyzer.h_bonds_history is None:
            self.h_bond_analyzer.find_hydrogen_bonds(
                box_size, bond_dict=bond_dict, focus_molecule_indices=focus_indices,
                distance_cutoff=h_bond_distance_cutoff, show_progress=show_progress,
                n_processes=n_processes
            )
        
        # Identify hydrogen bonds between focus group and environment
        inter_molecular_h_bonds = []
        
        for frame_idx, frame_h_bonds in enumerate(self.h_bond_analyzer.h_bonds_history):
            frame_inter_h_bonds = []
            
            for h_bond in frame_h_bonds:
                donor_idx = h_bond['donor_idx']
                acceptor_idx = h_bond['acceptor_idx']
                
                # Check if hydrogen bond connects focus group with environment
                if (donor_idx in focus_indices and acceptor_idx not in focus_indices) or \
                   (acceptor_idx in focus_indices and donor_idx not in focus_indices):
                    h_bond_with_frame = h_bond.copy()
                    h_bond_with_frame['frame'] = frame_idx
                    frame_inter_h_bonds.append(h_bond_with_frame)
            
            inter_molecular_h_bonds.append(frame_inter_h_bonds)
        
        # Calculate summary statistics
        total_frames = len(self.trajectory)
        
        covalent_interactions_per_frame = [len(frame_bonds) for frame_bonds in inter_molecular_bonds]
        h_bond_interactions_per_frame = [len(frame_h_bonds) for frame_h_bonds in inter_molecular_h_bonds]
        
        return {
            "inter_molecular_bonds": inter_molecular_bonds,
            "inter_molecular_h_bonds": inter_molecular_h_bonds,
            "avg_covalent_interactions": np.mean(covalent_interactions_per_frame),
            "max_covalent_interactions": np.max(covalent_interactions_per_frame),
            "avg_h_bond_interactions": np.mean(h_bond_interactions_per_frame),
            "max_h_bond_interactions": np.max(h_bond_interactions_per_frame),
            "frames_with_interactions": sum(1 for i in range(total_frames) 
                                          if len(inter_molecular_bonds[i]) > 0 or 
                                             len(inter_molecular_h_bonds[i]) > 0),
            "interaction_probability": sum(1 for i in range(total_frames) 
                                         if len(inter_molecular_bonds[i]) > 0 or 
                                            len(inter_molecular_h_bonds[i]) > 0) / total_frames
        }
    
    def get_solvation_shell(self, box_size, solute_indices, 
                          distance_cutoff=3.5, reference_frame=0,
                          include_h_bonds=True, show_progress=True):
        """
        Identify atoms in the solvation shell around a solute.
        
        Args:
            box_size: Simulation box dimensions with shape (3,)
            solute_indices: List of atom indices that make up the solute
            distance_cutoff: Maximum distance to consider an atom part of the solvation shell
            reference_frame: Frame index to use as reference
            include_h_bonds: Whether to include hydrogen bonding in the analysis
            show_progress: Whether to show progress bar
            
        Returns:
            dict: Solvation shell analysis results
        """
        if self.trajectory is None or self.elements_list is None:
            raise ValueError("No trajectory data. Call load_trajectory() or set_trajectory() first.")
            
        # Get reference coordinates and elements
        ref_coords = self.trajectory[reference_frame]
        ref_elements = self.elements_list[reference_frame]
        num_atoms = ref_coords.shape[0]
        
        # Calculate distances for reference frame with PBC
        solute_coords = ref_coords[solute_indices]
        
        # Solvation shell indices across all frames
        solvation_shell_history = []
        
        # Process each frame
        iterator = enumerate(self.trajectory)
        if show_progress:
            iterator = tqdm(iterator, total=len(self.trajectory), desc="Analyzing solvation shell")
            
        for frame_idx, coords in iterator:
            elements = self.elements_list[frame_idx]
            
            # Identify atoms in solvation shell based on distance
            solvation_indices = set()
            
            for solute_idx in solute_indices:
                solute_coord = coords[solute_idx]
                
                for atom_idx in range(num_atoms):
                    # Skip solute atoms
                    if atom_idx in solute_indices:
                        continue
                        
                    # Calculate distance with PBC
                    diff = coords[atom_idx] - solute_coord
                    diff = diff - box_size * np.round(diff / box_size)
                    distance = np.linalg.norm(diff)
                    
                    if distance <= distance_cutoff:
                        solvation_indices.add(atom_idx)
            
            # Add hydrogen bonds if requested
            if include_h_bonds and hasattr(self.h_bond_analyzer, 'h_bonds_history') \
               and self.h_bond_analyzer.h_bonds_history is not None:
                
                # Get hydrogen bonds for this frame
                frame_h_bonds = self.h_bond_analyzer.h_bonds_history[frame_idx]
                
                for h_bond in frame_h_bonds:
                    donor_idx = h_bond['donor_idx']
                    acceptor_idx = h_bond['acceptor_idx']
                    hydrogen_idx = h_bond['hydrogen_idx']
                    
                    # If the H-bond connects solute and solvent
                    if donor_idx in solute_indices and acceptor_idx not in solute_indices:
                        solvation_indices.add(acceptor_idx)
                    elif acceptor_idx in solute_indices and donor_idx not in solute_indices:
                        solvation_indices.add(donor_idx)
                        solvation_indices.add(hydrogen_idx)
            
            # Store solvation shell for this frame
            solvation_shell_history.append(list(solvation_indices))
        
        # Calculate summary statistics
        solvation_sizes = [len(shell) for shell in solvation_shell_history]
        
        # Get persistence of each atom in the solvation shell
        total_frames = len(self.trajectory)
        atom_persistence = {}
        
        for frame_idx, shell in enumerate(solvation_shell_history):
            for atom_idx in shell:
                if atom_idx not in atom_persistence:
                    atom_persistence[atom_idx] = 0
                atom_persistence[atom_idx] += 1
        
        # Convert counts to persistence ratios
        for atom_idx in atom_persistence:
            atom_persistence[atom_idx] /= total_frames
        
        # Sort atoms by persistence
        persistent_atoms = {k: v for k, v in 
                          sorted(atom_persistence.items(), key=lambda item: item[1], reverse=True)}
        
        return {
            "solvation_shell_history": solvation_shell_history,
            "avg_solvation_size": np.mean(solvation_sizes),
            "max_solvation_size": np.max(solvation_sizes),
            "min_solvation_size": np.min(solvation_sizes),
            "std_solvation_size": np.std(solvation_sizes),
            "atom_persistence": persistent_atoms
        }
    
    def export_trajectory_analysis(self, filename, analyzed_data=None, box_size=None):
        """
        Export analysis results to a file in JSON format.
        
        Args:
            filename: Path to output file
            analyzed_data: Optional pre-computed analysis data
            box_size: Simulation box dimensions needed if analyzed_data is not provided
            
        Returns:
            None
        """
        import json
        import numpy as np
        
        # Create a custom encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, set):
                    return list(obj)
                return super(NumpyEncoder, self).default(obj)
        
        # Perform analysis if not provided
        if analyzed_data is None:
            if box_size is None:
                raise ValueError("Box size must be provided if analyzed_data is not provided.")
                
            analyzed_data = self.analyze_trajectory(box_size)
        
        # Convert sets to lists for JSON serialization
        if "bond_history" in analyzed_data:
            analyzed_data["bond_history"] = [list(bonds) for bonds in analyzed_data["bond_history"]]
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(analyzed_data, f, cls=NumpyEncoder, indent=2)
            
        print(f"Analysis data exported to {filename}")
    def calculate_distance(self, atom_i, atom_j, frame=0, pbc=True, box_size=None):
        """
        Calculate distance between two atoms, with optional PBC handling.
        
        Args:
            atom_i: Index of first atom
            atom_j: Index of second atom
            frame: Frame number(s) to analyze - can be int or List[int] (default=0)
            pbc: Whether to apply periodic boundary conditions (default=True)
            box_size: Box dimensions as [Lx, Ly, Lz] for PBC
            
        Returns:
            float or list: Distance(s) between atoms in Angstroms
        """
        # Handle both single frame (int) and multiple frames (list)
        if isinstance(frame, int):
            frames = [frame]
            return_single = True
        else:
            frames = frame
            return_single = False
        
        results = []
        
        for f in frames:
            pos_i = self.trajectory[f][atom_i]
            pos_j = self.trajectory[f][atom_j]
            
            # Calculate distance vector
            dist_vec = pos_j - pos_i
            
            # Apply PBC if needed
            if pbc and box_size is not None:
                for dim in range(3):
                    if abs(dist_vec[dim]) > box_size[dim] / 2:
                        dist_vec[dim] -= np.sign(dist_vec[dim]) * box_size[dim]
            
            # Calculate the euclidean distance
            results.append(np.linalg.norm(dist_vec))
        
        # Return a single value if only one frame was specified
        return results[0] if return_single else results

    def calculate_dihedral(self, atom_i, atom_j, atom_k, atom_l, frame=0, pbc=True, box_size=None):
        """
        Calculate dihedral angle between four atoms, with optional PBC handling.
        
        Args:
            atom_i, atom_j, atom_k, atom_l: Indices of the four atoms defining the dihedral
            frame: Frame number(s) to analyze - can be int or List[int] (default=0)
            pbc: Whether to apply periodic boundary conditions (default=True)
            box_size: Box dimensions as [Lx, Ly, Lz] for PBC
            
        Returns:
            float or list: Dihedral angle(s) in degrees
        """
        # Handle both single frame (int) and multiple frames (list)
        if isinstance(frame, int):
            frames = [frame]
            return_single = True
        else:
            frames = frame
            return_single = False
        
        results = []
        
        for f in frames:
            # Get atom positions
            pos_i = self.trajectory[f][atom_i]
            pos_j = self.trajectory[f][atom_j]
            pos_k = self.trajectory[f][atom_k]
            pos_l = self.trajectory[f][atom_l]
            
            # Apply PBC to get minimum image vectors if needed
            if pbc and box_size is not None:
                # Calculate bond vectors with PBC
                rij = pos_j - pos_i
                rjk = pos_k - pos_j
                rkl = pos_l - pos_k
                
                for dim in range(3):
                    if abs(rij[dim]) > box_size[dim] / 2:
                        rij[dim] -= np.sign(rij[dim]) * box_size[dim]
                    if abs(rjk[dim]) > box_size[dim] / 2:
                        rjk[dim] -= np.sign(rjk[dim]) * box_size[dim]
                    if abs(rkl[dim]) > box_size[dim] / 2:
                        rkl[dim] -= np.sign(rkl[dim]) * box_size[dim]
                        
                # Recalculate positions with PBC
                pos_j = pos_i + rij
                pos_k = pos_j + rjk
                pos_l = pos_k + rkl
            
            # Calculate bond vectors
            b1 = pos_j - pos_i
            b2 = pos_k - pos_j
            b3 = pos_l - pos_k
            
            # Normalize bond vectors
            b2_norm = b2 / np.linalg.norm(b2)
            
            # Calculate normal vectors to the planes
            n1 = np.cross(b1, b2)
            n1 = n1 / np.linalg.norm(n1)
            n2 = np.cross(b2, b3)
            n2 = n2 / np.linalg.norm(n2)
            
            # Calculate the cosine and sine of the dihedral angle
            cos_phi = np.dot(n1, n2)
            sin_phi = np.dot(np.cross(n1, n2), b2_norm)
            
            # Calculate the dihedral angle
            phi = np.arctan2(sin_phi, cos_phi)
            
            # Convert to degrees
            results.append(np.degrees(phi))
        
        # Return a single value if only one frame was specified
        return results[0] if return_single else results

    def calculate_angle(self, atom_i, atom_j, atom_k, frame=0, pbc=True, box_size=None):
        """
        Calculate angle between three atoms, with optional PBC handling.
        
        Args:
            atom_i, atom_j, atom_k: Indices of the three atoms defining the angle
            frame: Frame number(s) to analyze - can be int or List[int] (default=0)
            pbc: Whether to apply periodic boundary conditions (default=True)
            box_size: Box dimensions as [Lx, Ly, Lz] for PBC
            
        Returns:
            float or list: Angle(s) in degrees
        """
        # Handle both single frame (int) and multiple frames (list)
        if isinstance(frame, int):
            frames = [frame]
            return_single = True
        else:
            frames = frame
            return_single = False
        
        results = []
        
        for f in frames:
            # Get atom positions
            pos_i = self.trajectory[f][atom_i]
            pos_j = self.trajectory[f][atom_j]
            pos_k = self.trajectory[f][atom_k]
            
            # Apply PBC to get minimum image vectors if needed
            if pbc and box_size is not None:
                # Calculate bond vectors with PBC
                rij = pos_j - pos_i
                rjk = pos_k - pos_j
                
                for dim in range(3):
                    if abs(rij[dim]) > box_size[dim] / 2:
                        rij[dim] -= np.sign(rij[dim]) * box_size[dim]
                    if abs(rjk[dim]) > box_size[dim] / 2:
                        rjk[dim] -= np.sign(rjk[dim]) * box_size[dim]
                        
                # Recalculate positions with PBC
                pos_j = pos_i + rij
                pos_k = pos_j + rjk
            
            # Calculate bond vectors
            vec1 = pos_i - pos_j
            vec2 = pos_k - pos_j
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate the cosine of the angle
            cos_angle = np.dot(vec1_norm, vec2_norm)
            
            # Ensure the value is within valid range for arccos
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Calculate the angle
            angle = np.arccos(cos_angle)
            
            # Convert to degrees
            results.append(np.degrees(angle))
        
        # Return a single value if only one frame was specified
        return results[0] if return_single else results

if __name__ == '__main__':
    analyzer = MolecularAnalyzer()
    analyzer.load_trajectory("./example.xyz")
    boxsize = [15.949258,15.949258,15.949258] 
    # Analyze the trajectory
    results = analyzer.analyze_trajectory(
        box_size=boxsize,
        covalent_bond_ratio=1.2,
        reference_frame=0,
        h_bond_distance_cutoff=4.0,
        min_h_bond_persistence=0.001,
        target_indices=None,
        n_processes=4,
        show_progress=True
    )
    
    # Print some summary information
    print("Covalent Bond Summary:")
    print(results["bond_summary"])    
    print("\nHydrogen Bond Statistics:")
    print(f"Average H-bonds per frame: {results['h_bond_stats']['avg_h_bonds_per_frame']:.2f}")
    print(f"Number of persistent H-bonds: {results['h_bond_stats']['persistent_h_bonds']}")
