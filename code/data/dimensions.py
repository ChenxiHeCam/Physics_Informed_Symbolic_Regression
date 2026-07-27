"""
semantic_class -> SI dimension vector, for feeding units/dimensions into the
data encoder. Dimensions are ALWAYS known at test time (we know what we measured),
unlike the abstract class itself — so this is a safe, robust input even for
unseen formulas, and dimensional analysis strongly constrains valid formulas.

Vector = 7 dims: [M, L, T, Theta, I, N, UNKNOWN_FLAG]
  M=mass L=length T=time Theta=temperature I=current N=amount, last = 1 if the
  class has no well-defined dimension (then the 6 powers are 0 and the model is
  told "unknown" rather than "dimensionless").
"""
# (M, L, T, Theta, I, N)
_D = {
    'length': (0, 1, 0, 0, 0, 0), 'wavelength': (0, 1, 0, 0, 0, 0),
    'velocity': (0, 1, -1, 0, 0, 0), 'acceleration': (0, 1, -2, 0, 0, 0),
    'mass': (1, 0, 0, 0, 0, 0), 'time': (0, 0, 1, 0, 0, 0),
    'temperature': (0, 0, 0, 1, 0, 0),
    'energy_generic': (1, 2, -2, 0, 0, 0), 'energy_kinetic': (1, 2, -2, 0, 0, 0),
    'energy_potential': (1, 2, -2, 0, 0, 0), 'energy_thermal': (1, 2, -2, 0, 0, 0),
    'gibbs_energy': (1, 2, -2, 0, 0, 0), 'chemical_potential': (1, 2, -2, 0, 0, 0),
    'hamiltonian': (1, 2, -2, 0, 0, 0), 'lagrangian': (1, 2, -2, 0, 0, 0),
    'torque': (1, 2, -2, 0, 0, 0),
    'force': (1, 1, -2, 0, 0, 0), 'surface_tension': (1, 0, -2, 0, 0, 0),
    'pressure': (1, -1, -2, 0, 0, 0), 'elastic_modulus': (1, -1, -2, 0, 0, 0),
    'stress_tensor': (1, -1, -2, 0, 0, 0), 'energy_density': (1, -1, -2, 0, 0, 0),
    'momentum': (1, 1, -1, 0, 0, 0),
    'angular_momentum': (1, 2, -1, 0, 0, 0), 'action': (1, 2, -1, 0, 0, 0),
    'power': (1, 2, -3, 0, 0, 0), 'luminosity': (1, 2, -3, 0, 0, 0),
    'heat_flux': (1, 0, -3, 0, 0, 0), 'intensity': (1, 0, -3, 0, 0, 0),
    'charge': (0, 0, 1, 0, 1, 0), 'current': (0, 0, 0, 0, 1, 0),
    'electric_potential': (1, 2, -3, 0, -1, 0),
    'electric_field': (1, 1, -3, 0, -1, 0),
    'magnetic_field': (1, 0, -2, 0, -1, 0),
    'magnetic_flux': (1, 2, -2, 0, -1, 0),
    'magnetic_moment': (0, 2, 0, 0, 1, 0),
    'resistance': (1, 2, -3, 0, -2, 0), 'impedance': (1, 2, -3, 0, -2, 0),
    'capacitance': (-1, -2, 4, 0, 2, 0), 'inductance': (1, 2, -2, 0, -2, 0),
    'conductance': (-1, -2, 3, 0, 2, 0),
    'permittivity': (-1, -3, 4, 0, 2, 0), 'permeability': (1, 1, -2, 0, -2, 0),
    'charge_density': (0, -3, 1, 0, 1, 0), 'current_density': (0, -2, 0, 0, 1, 0),
    'electric_dipole_moment': (0, 1, 1, 0, 1, 0),
    'frequency': (0, 0, -1, 0, 0, 0), 'angular_frequency': (0, 0, -1, 0, 0, 0),
    'decay_rate': (0, 0, -1, 0, 0, 0), 'reaction_rate': (0, 0, -1, 0, 0, 0),
    'hubble_param': (0, 0, -1, 0, 0, 0),
    'wavenumber': (0, -1, 0, 0, 0, 0),
    'area': (0, 2, 0, 0, 0, 0), 'cross_section': (0, 2, 0, 0, 0, 0),
    'volume': (0, 3, 0, 0, 0, 0),
    'mass_density_generic': (1, -3, 0, 0, 0, 0), 'fluid_density': (1, -3, 0, 0, 0, 0),
    'gas_density': (1, -3, 0, 0, 0, 0), 'charge_density_mass': (1, -3, 0, 0, 0, 0),
    'number_density': (0, -3, 0, 0, 0, 0), 'density_of_states': (0, -3, 0, 0, 0, 0),
    'density_states': (0, -3, 0, 0, 0, 0),
    'surface_density': (1, -2, 0, 0, 0, 0), 'moment_of_inertia': (1, 2, 0, 0, 0, 0),
    'mass_flow_rate': (1, 0, -1, 0, 0, 0),
    'entropy': (1, 2, -2, -1, 0, 0), 'specific_heat': (1, 2, -2, -1, 0, 0),
    'thermal_conductivity': (1, 1, -3, -1, 0, 0),
    'viscosity': (0, 2, -1, 0, 0, 0), 'diffusion_coeff': (0, 2, -1, 0, 0, 0),
    'concentration': (0, -3, 0, 0, 0, 1),
    # explicitly dimensionless
    'dimensionless_ratio': (0, 0, 0, 0, 0, 0), 'angle': (0, 0, 0, 0, 0, 0),
    'solid_angle': (0, 0, 0, 0, 0, 0), 'refractive_index': (0, 0, 0, 0, 0, 0),
    'strain': (0, 0, 0, 0, 0, 0), 'spin': (0, 0, 0, 0, 0, 0),
    'count_integer': (0, 0, 0, 0, 0, 0), 'atomic_number': (0, 0, 0, 0, 0, 0),
    'mass_number': (0, 0, 0, 0, 0, 0), 'redshift': (0, 0, 0, 0, 0, 0),
    'susceptibility': (0, 0, 0, 0, 0, 0), 'magnitude': (0, 0, 0, 0, 0, 0),
    'metric_tensor': (0, 0, 0, 0, 0, 0), 'activity_coeff': (0, 0, 0, 0, 0, 0),
    'partition_function': (0, 0, 0, 0, 0, 0), 'compressibility': (-1, 1, 2, 0, 0, 0),
}
# genuinely ambiguous -> UNKNOWN flag (powers 0, flag 1)
_UNKNOWN = {
    'other', 'fundamental_constant', 'wavefunction', 'probability_amplitude',
    'christoffel', 'ricci', 'polarization', 'magnetization', 'electric_flux',
    'heat_flux_vector', 'tensor', 'field_generic',
}

DIM_LEN = 7  # 6 SI powers + unknown flag


def class_to_dim(cls):
    if cls in _D:
        return tuple(_D[cls]) + (0,)
    return (0, 0, 0, 0, 0, 0, 1)   # unknown


def dims_for(symbols, sym2class):
    """symbols -> (len(symbols), 7) list of dimension vectors via semantic_class."""
    return [class_to_dim(sym2class.get(s, 'other')) for s in symbols]


if __name__ == '__main__':
    for c in ['velocity', 'energy_generic', 'electric_potential', 'other', 'mass']:
        print(c, class_to_dim(c))
