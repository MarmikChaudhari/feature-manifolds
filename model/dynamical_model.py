"""
Dynamical System Model for N-Sphere Particle Simulation

Simulates N points on a unit (n-1)-sphere in R^n with pairwise attraction/repulsion forces.
Particles attract neighbors within a zone width and repel distant points, with dynamics
constrained to the sphere manifold.
"""

import numpy as np
from typing import Literal


class SphereDynamics:
    """
    Dynamical system for N particles on an (n-1)-dimensional sphere.
    
    Parameters:
    n_particles : int
        Number of particles (default: 100)
    n_dimensions : int
        Ambient dimension ({3, 4, 5, 6, 7, 8})
    zone_width : float
        Attractive zone width parameter w (default: 5)
    topology : str
        Either 'circle' or 'interval' for index distance computation
    dt : float
        Time step size (default: 0.01)
    damping : float
        Velocity damping coefficient alpha (default: 0.95)
    damping_linear : float
        Linear damping term in force equation (default: 0.05)
    """
    
    def __init__(
        self,
        n_particles: int = 100,
        n_dimensions: int = 6,
        zone_width: float = 5.0,
        topology: Literal['circle', 'interval'] = 'circle',
        dt: float = 0.01,
        damping: float = 0.95,
        damping_linear: float = 0.05
    ):
        if n_dimensions not in {3, 4, 5, 6, 7, 8}:
            raise ValueError("Dimension must be in {3, 4, 5, 6, 7, 8}")
        
        self.N = n_particles
        self.n_dims = n_dimensions
        self.w = zone_width
        self.topology = topology
        self.dt = dt
        self.damping = damping
        self.damping_linear = damping_linear
        
        # initialize positions on sphere and velocities
        self.positions = self._initialize_sphere()
        self.velocities = np.zeros((self.N, self.n_dims))
        
    def _initialize_sphere(self) -> np.ndarray:
        """
        initialize N points uniformly distributed on the unit (n-1)-sphere in R^n.
        
        returns:
        positions : np.ndarray of shape (N, n_dims) - points on the unit sphere
        """
        # sample from standard normal distribution and normalize
        positions = np.random.randn(self.N, self.n_dims)
        return self._normalize_to_sphere(positions)
    
    def _normalize_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """
        project points onto the unit sphere: x_i <- x_i / ||x_i||
            
        returns:
        normalized : np.ndarray of shape (N, n_dims) - points normalized to unit sphere
        """
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return points / norms
    
    def _compute_index_distances(self) -> np.ndarray:
        """
        compute index distances d_ij based on topology.
        
        for circular topology: d_ij = min(|j-i|, |j-i+N|, |j-i-N|)
        for interval topology: d_ij = |j-i|
        
        returns:
        index_distances : np.ndarray of shape (N, N) - index distance matrix
        """
        # create index arrays
        i_indices = np.arange(self.N).reshape(-1, 1)
        j_indices = np.arange(self.N).reshape(1, -1)
        
        if self.topology == 'circle':
            # Circular topology: consider wraparound
            diff = j_indices - i_indices
            distances = np.minimum(
                np.abs(diff),
                np.minimum(
                    np.abs(diff + self.N),
                    np.abs(diff - self.N)
                )
            )
        else:  # interval
            # Interval topology: simple absolute difference
            distances = np.abs(j_indices - i_indices)
        
        return distances
    
    def _compute_spatial_distances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        compute spatial distances and direction vectors between all pairs of points.
        
        returns:
        r_ij : np.ndarray of shape (N, N) - Euclidean distances ||x_j - x_i||
        r_hat_ij : np.ndarray of shape (N, N, n_dims) - unit direction vectors (x_j - x_i) / r_ij
        diff : np.ndarray of shape (N, N, n_dims) - position differences x_j - x_i
        """
        # compute pairwise differences: x_j - x_i
        diff = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        
        # compute distances r_ij = ||x_j - x_i||
        r_ij = np.linalg.norm(diff, axis=2)
        
        # avoid division by zero (self-interaction will be masked later)
        r_ij_safe = np.where(r_ij > 1e-10, r_ij, 1.0)
        
        # compute unit direction vectors r̂_ij = (x_j - x_i) / r_ij
        r_hat_ij = diff / r_ij_safe[:, :, np.newaxis]
        
        return r_ij, r_hat_ij, diff
    
    def _compute_forces(self) -> np.ndarray:
        """
        compute pairwise forces F_ij for all particle pairs.
        
        force law:
            F_ij = (1-(d_ij-1)/2) / r_ij * r̂_ij    when d_ij ≤ w (attractive)
            F_ij = -min(5, 1/r_ij) / r_ij * r̂_ij   when d_ij > w (repulsive)
        
        returns:
        forces : np.ndarray of shape (N, n_dims) - net force on each particle (sum over j≠i)
        """
        # compute index distances d_ij
        d_ij = self._compute_index_distances()
        
        # compute spatial distances and directions
        r_ij, r_hat_ij, _ = self._compute_spatial_distances()
        
        # compute force magnitudes for each case
        # case 1: d_ij ≤ w (attractive force)
        attractive_magnitude = (1 - (d_ij - 1) / 2) / np.where(r_ij > 1e-10, r_ij, 1.0)
        
        # case 2: d_ij > w (repulsive force)
        repulsive_magnitude = -np.minimum(5.0, 1.0 / np.where(r_ij > 1e-10, r_ij, 1.0)) / np.where(r_ij > 1e-10, r_ij, 1.0)
        
        # select magnitude based on condition d_ij ≤ w
        force_magnitude = np.where(d_ij <= self.w, attractive_magnitude, repulsive_magnitude)
        
        # compute force vectors F_ij = magnitude * r_hat_ij
        F_ij = force_magnitude[:, :, np.newaxis] * r_hat_ij
        
        # mask self-interaction (i.e., F_ii = 0)
        mask = np.eye(self.N, dtype=bool)
        F_ij[mask] = 0
        
        # sum forces over all j≠i for each particle i
        forces = np.sum(F_ij, axis=1)
        return forces
    
    def step(self) -> None:
        """
        one time step of the dynamics.
        
        evolution equations:
            v_dot_i = Σ_{j≠i} F_ij - damping_linear * v_i
            x_dot_i = v_i
        """
        # compute net forces on all particles
        forces = self._compute_forces()
        # update velocities: v_dot_i = Σ F_ij - damping_linear * v_i
        dv_dt = forces - self.damping_linear * self.velocities
        self.velocities += dv_dt * self.dt
        # apply velocity damping: v_i <- α * v_i
        self.velocities *= self.damping
        # update positions: x_i <- x_i + v_i * dt
        self.positions += self.velocities * self.dt
        # enforce sphere constraint: x_i <- x_i / ||x_i||
        self.positions = self._normalize_to_sphere(self.positions)
    
    def simulate(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        run simulation for n_steps time steps.
        
        returns:
        trajectory : np.ndarray of shape (n_steps, N, n_dims) - position history
        velocity_history : np.ndarray of shape (n_steps, N, n_dims) - velocity history
        """
        trajectory = np.zeros((n_steps, self.N, self.n_dims))
        velocity_history = np.zeros((n_steps, self.N, self.n_dims))
        
        for t in range(n_steps):
            trajectory[t] = self.positions.copy()
            velocity_history[t] = self.velocities.copy()
            self.step()
        
        return trajectory, velocity_history
    
    def reset(self) -> None:
        """reset the system to a new random initial configuration."""
        self.positions = self._initialize_sphere()
        self.velocities = np.zeros((self.N, self.n_dims))
    
    
    def get_inner_product_matrix(self) -> np.ndarray:
        """
        compute the inner product matrix <x_i, x_j> for all particle pairs.
                
        returns:
        inner_products : np.ndarray of shape (N, N) - matrix of inner products
        """
        return self.positions @ self.positions.T