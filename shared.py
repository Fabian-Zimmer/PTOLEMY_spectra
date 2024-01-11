# Default libraries
import sys

# Conda install libraries
import numpy as np
import jax
import jax.scipy as jsp
import jax.numpy as jnp
import chex
import astropy.units as apu
import astropy.constants as apc
from scipy.special import zeta
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from icecream import ic
import healpy

# Pip install libraries
import diffrax
import natpy as nat

# Switch jax to use float64 as default instead of float32
from jax.config import config
config.update("jax_enable_x64", True)


@chex.dataclass
class Params:
    #############
    ### Units ###
    #############
    GB_UNIT: float = 1000*1024**2
    Pi: float      = jnp.pi
    hc_val: float  = (apc.h/(2*Pi)*apc.c).convert(apu.eV*apu.cm).value

    eV: float     = 1.                         # Unit of energy: eV
    meV: float    = 1.0e-3*eV
    keV: float    = 1.0e3*eV
    MeV: float    = 1.0e3*keV
    GeV: float    = 1.0e3*MeV
    TeV: float    = 1.0e3*GeV
    erg: float    = TeV/1.602                  # erg
    J: float      = 1.0e7*erg                  # joule
    K: float      = 8.61732814974493e-5*eV     # Kelvin

    cm: float     = (1/hc_val)/eV              # centi-meter
    m: float      = 1.0e2*cm
    km: float     = 1.0e3*m
    pc: float     = 3.08567758128e18*cm        # parsec
    kpc: float    = 1.0e3*pc
    Mpc: float    = 1.0e3*kpc

    # note: 
    # Natural units defined via c=1, i.e. s/m = 299792458
    s: float      = 2.99792458e10*cm           # second
    yr: float     = 365*24*60*60*s
    Gyr: float    = 1e9*yr
    t0: float     = 13.787*Gyr
    Hz: float     = 1.0/s
    c: float      = 299792458*m/s

    kg: float     = J/m**2*s**2
    gram: float   = kg/1000.
    Msun: float   = 1.98847e30*kg              # Mass of the Sun
    G: float      = 6.6743e-11*m**3/kg/s**2    # Gravitational constant
    Da: float     = 1.66053906660e-27*kg       # Dalton or atomic mass unit (u)

    deg: float    = Pi/180.0                   # Degree
    arcmin: float = deg/60.                    # Arcminute
    arcsec: float = arcmin/60.                 # Arcsecond
    sr: float     = 1.                         # Steradian


    #################
    ### Constants ###
    #################
    # Cosmological parameters    
    h: float       = 0.674
    H0: float      = h * 100 * km/s/Mpc
    Omega_R: float = 9.23640e-5
    Omega_M: float = 0.3111
    Omega_L: float = 1.-Omega_M-Omega_R

    # Temperatures of CMB and CNB
    T_CMB: float = 2.72548*K
    T_CNB = jnp.power(4/11, 1/3)*T_CMB

    # Redshift to last-scattering surfaces of CMB and CNB
    Z_LSS_CMB: float = 1100.0
    Z_LSS_CNB: float = 6.e9

    # Neutrino degrees of freedom (d.o.f.)
    g_nu: float = 1.

    # Cosmological (i.e. LambdaCDM) relic neutrino number density, per d.o.f.
    N0 = zeta(3.)/Pi**2 * T_CNB**3 * (3./4.)

    # Key to fix random numbers of code
    key = jax.random.PRNGKey(0)

    #############################
    ### Simulation Parameters ###
    #############################
    p_start: float = 0.01
    p_stop: float  = 400.0
    p_num: int     = 200
    phis: int      = 20
    thetas: int    = 20


    ##########################
    ### PTOLEMY Parameters ###
    ##########################
    """From https://arxiv.org/abs/1902.05508."""
    sigma_avg: float = 3.834e-45*cm**2   # velocity-averaged cross-section
    M_T: float       = 100*gram          # target total Tritium mass in setup
    M_3H: float      = 2809.432*MeV      # atomic mass of Helium-3 (3He)
    M_3He: float     = 2809.413*MeV      # atomic mass of 3H (Tritium)
    m_3H: float      = 2808.921*MeV      # nuclear mass of 3H (Tritium)
    m_3He: float     = 2808.391*MeV      # nuclear mass of Helium-3 (3He)
    N_T: float       = M_T / m_3H        # number of Tritium atoms

    m_e: float       = 0.510998950*MeV   # electron mass
    m_n: float       = 939.56542052*MeV  # neutron mass
    m_p: float       = 938.27208816*MeV  # proton mass

    # Kinetic energy of electron at beta-decay endpoint, with a massless neutrino.
    """Lower eqn. of (5.5) in http://arxiv.org/abs/2109.02900."""
    K0_end: float = ((m_3H - m_e)**2 - m_3He**2)/(2*m_3H)


    ###########################
    ### Neutrino Parameters ###
    ###########################
    # Taken from https://arxiv.org/abs/2007.14792

    # PMNS matrix elements of electron flavor (absolute value squared), i.e.
    # |U_e1|^2, |U_e2|^2, |U_e3|^2
    # Taken as average values between upper and lower limits from eqn. (4.2)

    # For Normal Ordering
    U_ei_AbsSq_NO = jnp.array([0.681, 0.297, 0.0222])

    # For Inverted Ordering
    U_ei_AbsSq_IO = jnp.array([0.0222, 0.681, 0.297])

    # Measured neutrino mass squared differences from Table 3 (witout SK atm. data)
    Del_m21_Sq: float = 7.42e-5 * eV**2
    Del_m31_Sq_NO: float = 2.514e-3 * eV**2
    Del_m32_Sq_IO: float = 2.497e-3 * eV**2


    #############
    ### Plots ###
    #############
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('figure', figsize=(8, 8))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('xtick', direction='in', top=True)  # x-axis ticks inside, top enabled
    plt.rc('ytick', direction='in', right=True) # y-axis ticks inside, right enabled


@chex.dataclass
class Data:

    @staticmethod
    def check_array_size_GB(array):

        # Get the size of each element in bytes
        element_size_bytes = array.itemsize

        # Get the total number of elements
        total_elements = array.size

        # Calculate the total size in bytes
        total_size_bytes = element_size_bytes * total_elements

        # Convert the total size to gigabytes (GB)
        total_size_gb = total_size_bytes / (1024 ** 3)

        print(f'Memory size is {total_size_gb:.3f} GB')


    @staticmethod
    def load_neutrino_velocities(halo_ID:str):

        file_name = f'neutrino_vectors_numerical_{halo_ID}'

        nu_vels = jnp.array([
            jnp.load(f'Data/{file_name}_batch{i+1}.npy')[...,3:6]
            for i in range(8)
        ]).reshape(-1,100,3)

        return nu_vels


@chex.dataclass
class Physics:

    """
    Cosmological functions.
    """

    @staticmethod
    @jax.jit
    def Hubble_z(z, args):
        """Eqn. (4.8)."""
        return args.H0*jnp.sqrt(args.Omega_M*(1.+z)**3 + args.Omega_L)
    

    @staticmethod
    @jax.jit
    def time_z(z, args):
        """
        Time of the universe at redshift z, when dark matter and dark energy are 
        dominating and rest is negligible. Eqn. (4.9).
        """
        r_z = args.Omega_L/args.Omega_M * 1/(1.+z)**3
        return 2/(3*args.H0*jnp.sqrt(args.Omega_L)) * jnp.log(jnp.sqrt(r_z) + jnp.sqrt(1+r_z))
    

    @staticmethod
    def neutrino_masses(m_lightest, ordering, args):
        """Returns the 3 neutrino masses with given ordering and lightest mass."""

        def normal_ordering(_):
            m1 = m_lightest
            m2 = jnp.sqrt(m1**2 + args.Del_m21_Sq)
            m3 = jnp.sqrt(m1**2 + args.Del_m31_Sq_NO)
            return jnp.array([m1, m2, m3])

        def inverted_ordering(_):
            m3 = m_lightest
            m2 = jnp.sqrt(m3**2 + args.Del_m32_Sq_IO)
            m1 = jnp.sqrt(m2**2 - args.Del_m21_Sq)
            return jnp.array([m3, m1, m2])

        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")

        # Switch based on the ordering
        branches = [normal_ordering, inverted_ordering]
        return jax.lax.switch(ordering_index[ordering], branches, None)


    @staticmethod
    @jax.jit
    def velocities_to_momenta(v_arr, m_arr, args):
        """
        Converts velocities to magnitude of momentum [eV] and ratio y=p/T_CNB, 
        according to desired target mass (and mass used in simulation).
        """

        # Magnitudes of velocity.
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        mags_dim = jnp.repeat(jnp.expand_dims(mags_arr, axis=0), len(m_arr), axis=0)

        # Adjust neutrino target mass array dimensionally.
        m_dim = jnp.expand_dims(
            jnp.repeat(
                jnp.expand_dims(m_arr, axis=1), mags_dim.shape[1], axis=1),
            axis=2)

        # From velocity (magnitude) in kpc/s to momentum in eV.
        p_dim = mags_dim*(args.kpc/args.s) * m_dim

        # p/T_CNB ratio.
        y = p_dim/args.T_CNB

        return p_dim, y
    

    @staticmethod
    def filter_momenta(p_arr, y_arr, m_len, args):

        # Select necessary sub-arrays
        p_init = p_arr[...,0]
        p_back = p_arr[...,-1]
        y_init = y_arr[...,0]

        # Sort
        sort_indices = p_init.argsort(axis=-1)
        p_back_sort = jnp.take_along_axis(p_back, sort_indices, axis=-1)
        y_init_sort = jnp.take_along_axis(y_init, sort_indices, axis=-1)

        # Select "most likely", i.e. most clustered neutrinos, of each velocity batch
        p_back_blocks = p_back_sort.reshape((m_len, args.p_num, args.phis*args.thetas))
        p_back_select = jnp.min(p_back_blocks, axis=-1)
        y_init_blocks = y_init_sort.reshape((m_len, args.p_num, args.phis*args.thetas))
        y_init_select = y_init_blocks[...,0]

        return p_back_select, y_init_select
    

    @staticmethod
    @jax.jit
    def Fermi_Dirac(p, args):
        return 1/(jnp.exp(p/args.T_CNB) + 1)
    

    @staticmethod
    def Energy_Momentum_conversion(X_arr, mass_arr, mode='E_to_p'):

        def Energy_to_momentum(_):
            signs = jnp.sign(X_arr[None,:])
            diffs = jnp.sqrt(X_arr[None, :]**2 - mass_arr[:, None]**2)
            return signs * diffs
        
        def Momentum_to_energy(_):
            signs = jnp.sign(X_arr[None,:])
            diffs = jnp.sqrt(X_arr[None, :]**2 + mass_arr[:, None]**2)
            return signs * diffs

        # Map modes to indices
        mode_index = {'E_to_p': 0, 'p_to_E': 1}
        if mode not in mode_index:
            raise ValueError("Invalid 'ordering' value. It should be 'E_to_p' or 'p_to_E'.")

        # Switch based on the mode
        branches = [Energy_to_momentum, Momentum_to_energy]
        return jax.lax.switch(mode_index[mode], branches, None)


@chex.dataclass
class Ptolemy:

    """
    The Equations are referring to the Akita et al. (2022) paper:
    http://arxiv.org/abs/2109.02900
    """

    @staticmethod
    @jax.jit
    def Gamma_i_to_j(m_i, lambda_ij, args):
        """Eqn. (3.3)."""
        return 1/args.t0 * m_i/(50*args.meV) * (lambda_ij/6.2e-16)**2


    @staticmethod
    @jax.jit
    def n_i_today(m_i, lambda_ij, f_c, n0_i, args):
        """Eqn. (3.4)"""
        return jnp.exp(- m_i/(50*args.meV) * (lambda_ij/6.2e-16)**2) * f_c * n0_i
    

    @staticmethod
    @jax.jit
    def p_j_z(p_j, z):
        """Upper eqn. of (4.6)."""
        return p_j*(1.+z)
    

    @staticmethod
    @jax.jit
    def E_j_z(p_j, m_j, z):
        """Lower eqn. of (4.6)."""
        return jnp.sqrt(p_j**2*(1.+z)**2 + m_j**2)


    @staticmethod
    @jax.jit
    def p_star(m_i, m_j, m_x):
        """Upper eqn. of (4.11)."""
        return 1/2/m_i*jnp.sqrt((m_i**2 - (m_j + m_x)**2)*(m_i**2 - (m_j - m_x)**2))


    @staticmethod
    @jax.jit
    def E_star(m_i, m_j, m_x):
        """Lower eqn. of (4.11)."""
        return (m_i**2 + m_j**2 - m_x**2)/(2*m_i)


    @staticmethod
    @jax.jit
    def E_j_from_E_e(E_e, args):
        """Upper eqn. of (5.5)."""
        return E_e - args.K0_end - args.m_e


    @staticmethod
    @jax.jit
    def E_e_from_E_j(E_j, args):
        """Inverse of upper eqn. of (5.5)."""
        return E_j + args.K0_end + args.m_e


    @staticmethod
    @jax.jit
    def Ei_end(m_i, args):
        """Eqn. (5.9)"""
        return args.K0_end + args.m_e - m_i


    @staticmethod
    @jax.jit
    def E_end(m_lightest, args):
        """Eqn. (5.10)"""
        return args.K0_end + args.m_e - m_lightest

    @staticmethod
    @jax.jit
    def Q_value(m_nu, args):
        return args.m_3H - args.m_3He - args.m_e - m_nu


    @staticmethod
    @jax.jit
    def K_end(m_nu, args):
        return ((args.m_3H - args.m_e)**2 - (m_nu - args.m_3He)**2)/(2*args.m_3H)


    @staticmethod
    @jax.jit
    def H_factor(E_e, m_i, args):
        """Eqn. (5.8)"""
        
        numer1 = 1 - (args.m_e**2/(E_e*args.m_3H))
        denom1 = (1 - (2*E_e/args.m_3H) + (args.m_e**2/args.m_3H**2))**2
        term1 = numer1/denom1
        
        # set negative values to zero
        y_i = jnp.clip(Ptolemy.Ei_end(m_i, args) - E_e, 0., None)
        
        term2 = jnp.sqrt(y_i * (y_i + (2*m_i*args.m_3He/args.m_3H)))
        term3 = y_i + m_i/args.m_3H*(args.m_3He + m_i)
        
        return term1*term2*term3
        

    @staticmethod
    def Beta_decay_spectrum(E_e, nu_masses, ordering, args):
        """Eqn. (5.7)"""

        # H factors for each neutrino mass.
        H_vals = jnp.array([Ptolemy.H_factor(E_e, m_i, args) for m_i in nu_masses])

        # dGamma_beta/dE_e.
        def normal_ordering(_):
            return jnp.tile(args.U_ei_AbsSq_NO, (len(E_e),1)).T
        
        def inverted_ordering(_):
            return jnp.tile(args.U_ei_AbsSq_IO, (len(E_e),1)).T

        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")

        branches = [normal_ordering, inverted_ordering]
        U_ei_tiled = jax.lax.switch(ordering_index[ordering], branches, None)

        return args.sigma_avg/args.Pi**2*args.N_T * jnp.sum(U_ei_tiled*H_vals, axis=0)


    @staticmethod
    @jax.jit
    def dndE_decay(E_j, m_j, m_i, m_x, tau_i, n0_i, args):
        """Eqn. (4.13)."""

        def true_branch(_):
            return jnp.array(0.0)

        def false_branch(_):
            zE = (Ptolemy.p_star(m_i, m_j, m_x) / jnp.sqrt(E_j**2 - m_j**2)) - 1
            tE = Physics.time_z(zE, args)
            const_term = n0_i * jnp.exp(-tE / tau_i) / (Physics.Hubble_z(zE, args) * tau_i)
            energy_term = E_j / (E_j**2 - m_j**2)
            return const_term * energy_term

        E_star = Ptolemy.E_star(m_i, m_j, m_x)
        condition = (E_j <= m_j) | (E_j > E_star)

        return jax.lax.cond(condition, true_branch, false_branch, None)


    @staticmethod
    def dGammaTilde_CNB_dE_e(E_e, m_j, m_i, m_x, tau_i, n0_i, ordering, args):
        """Eqn. (5.6)"""

        def normal_ordering(_):
            return 2*args.U_ei_AbsSq_NO[0]*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return 2*args.U_ei_AbsSq_IO[0]*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)

        E_j = Ptolemy.E_j_from_E_e(E_e, args)
        dnj_dEj = jnp.array([
            Ptolemy.dndE_decay(E, m_j, m_i, m_x, tau_i, n0_i, args) for E in E_j
        ])

        return pre*dnj_dEj
    

    @staticmethod
    def dndE_Fermi_Dirac(nu_masses, p_start, p_stop, p_num, args):

        # Momentum range determined by input parameters
        p_FD = jnp.geomspace(p_start, p_stop, p_num)
        
        # Compute Fermi-Dirac values
        f_FD = Physics.Fermi_Dirac(p_FD, args)

        # Neutrino number per momentum interval
        dndp_FD = p_FD**2 * f_FD

        # Get corresponding energy range
        E_FD = Physics.Energy_Momentum_conversion(p_FD, nu_masses, 'p_to_E')

        # Get velocities via momentum-energy relation
        v_FD = p_FD / E_FD

        # Neutrino number per energy interval
        E_sqrt = jnp.sqrt(E_FD**2 - nu_masses[:,None]**2)
        dndE_FD = E_FD * E_sqrt * f_FD

        return p_FD, dndp_FD, E_FD, dndE_FD, v_FD


    @staticmethod
    def dGamma_CNB_dE_e_FD(nu_masses, ordering, nature, p_start, p_stop, p_num, args):

        def normal_ordering(_):
            return nature*args.U_ei_AbsSq_NO*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return nature*args.U_ei_AbsSq_IO*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        # Choose ordering and determine appropriate prefactors
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)
        phase_space_factor = args.g_nu / (2 * args.Pi**2)

        # Get Fermi-Dirac energy spectra
        *_, E_FD, dndE_FD, _ = Ptolemy.dndE_Fermi_Dirac(
            nu_masses, p_start, p_stop, p_num, args)

        return E_FD, phase_space_factor * pre[:,None] * dndE_FD
    

    @staticmethod
    def dndE_simulation(halo_ID, nu_masses, args):

        # Load velocity data
        v_sim = Data.load_neutrino_velocities(halo_ID) * (args.kpc/args.s**2)

        # Convert velocities to momenta
        p_sim, _ = Physics.velocities_to_momenta(v_sim, nu_masses, args)

        # Compute Fermi-Dirac values
        f_sim = Physics.Fermi_Dirac(p_sim, args)

        # Neutrino number per momentum interval
        dndp_sim = p_sim**2 * f_sim

        # Get corresponding energy range
        E_sim = Physics.Energy_Momentum_conversion(p_sim, nu_masses, 'p_to_E')

        # Neutrino number per energy interval
        E_sqrt = jnp.sqrt(E_sim**2 - nu_masses[:,None]**2)
        dndE_sim = E_sim * E_sqrt * f_sim

        return p_sim, dndp_sim, E_sim, dndE_sim
    

    @staticmethod
    def dGamma_CNB_dE_e_simulation(halo_ID, nu_masses, ordering, nature, args):

        def normal_ordering(_):
            return nature*args.U_ei_AbsSq_NO*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return nature*args.U_ei_AbsSq_IO*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        # Choose ordering and determine appropriate prefactors
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)
        phase_space_factor = args.g_nu / (2 * args.Pi**2)

        # Get CNB simulation energy spectra
        *_, E_sim, dndE_sim, _ = Ptolemy.dndE_simulation(
            halo_ID, nu_masses, args)

        return E_sim, phase_space_factor * pre[:,None] * dndE_sim


    @staticmethod
    def analyze_peak_for_X_range_resolution(data, points_in_FWHM):
        # Finding the peak
        peak_value = jnp.max(data)
        peak_idx = jnp.argmax(data)

        # Calculating the Full Width at Half Maximum (FWHM)
        half_max = peak_value / 2
        left_idx = jnp.where(data[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data[peak_idx:] < half_max)[0][0] + peak_idx
        FWHM = right_idx - left_idx

        # Determining the resolution for the linear x-axis
        total_range = len(data)
        resolution = int((total_range / FWHM) * points_in_FWHM) + 1

        return resolution, peak_idx, FWHM


    @staticmethod
    def analyze_peak_for_energy_step(data_x, data_y, points_in_FWHM):
        """
        Analyzes the peak in the spectrum to determine the energy step size for a given resolution.

        :param data_x: The original energy values (x-axis).
        :param data_y: The corresponding spectrum values (y-axis).
        :param points_in_FWHM: The number of points desired within the FWHM.
        :return: The necessary energy step size.
        """
        # Finding the peak
        peak_value = jnp.max(data_y)
        peak_idx = jnp.argmax(data_y)

        # Calculating the Full Width at Half Maximum (FWHM)
        half_max = peak_value / 2
        left_idx = jnp.where(data_y[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data_y[peak_idx:] < half_max)[0][0] + peak_idx

        # Energy values at FWHM
        E_FWHM_left = data_x[left_idx]
        E_FWHM_right = data_x[right_idx]

        # Determining the energy step size
        FWHM_energy_range = E_FWHM_right - E_FWHM_left
        energy_step = FWHM_energy_range / points_in_FWHM

        return energy_step, E_FWHM_left, E_FWHM_right


    @staticmethod
    def analyze_peak_and_determine_extended_resolution(data, points_in_FWHM, E_start_orig, E_stop_orig, E_min, E_max):
        """
        Determines the resolution for an extended linear energy range based on the FWHM of the original data.

        :param data: The array containing the data.
        :param points_in_FWHM: The number of points desired within the FWHM.
        :param E_start_orig: The starting energy of the original data range.
        :param E_stop_orig: The stopping energy of the original data range.
        :param E_min: The minimum energy of the extended range.
        :param E_max: The maximum energy of the extended range.
        :return: The number of points for the extended linear energy range.
        """
        # Finding the peak and FWHM in the original data
        peak_value = jnp.max(data)
        peak_idx = jnp.argmax(data)
        half_max = peak_value / 2
        left_idx = jnp.where(data[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data[peak_idx:] < half_max)[0][0] + peak_idx
        FWHM = right_idx - left_idx

        # Original energy range and step size
        original_range = E_stop_orig - E_start_orig
        step_size = original_range / FWHM

        # Number of points in the original range to achieve the desired resolution
        num_points_original = int((original_range / step_size) * points_in_FWHM)

        # Applying the same step size to the extended range
        extended_range = E_max - E_min
        num_points_extended = int(extended_range / step_size) + 1

        return num_points_extended, num_points_original, FWHM



    @staticmethod
    def gpr_interpolation(x_original, y_original, x_interpolated):
        """
        Interpolates irregularly spaced data using Gaussian Process Regression.
        """
        
        # Reshaping for compatibility with GPR
        x_original_reshaped = jnp.array(x_original).reshape(-1, 1)
        x_interpolated_reshaped = jnp.array(x_interpolated).reshape(-1, 1)

        # Matérn kernel with a specified nu
        kernel = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Fitting the model
        gp.fit(x_original_reshaped, y_original)

        # Predicting the interpolated data
        y_interpolated, sigma = gp.predict(x_interpolated_reshaped, return_std=True)
        
        return y_interpolated, sigma
    

    @staticmethod
    @jax.jit
    def Gaussian_kernel(X_range, mu, sigma, args):

        # Normalization for Gaussian
        norm = jnp.sqrt(2*args.Pi)*sigma  

        return jnp.exp(-(X_range - mu)**2 / (2 * sigma**2)) / norm


    @staticmethod
    @jax.jit
    def Gaussian_convolution_1D(X_range, array_to_convolve, Gaussian_sigma, args):

        # X-range differences for the Gaussian in the convolution integral
        X_diff = X_range[:, None] - X_range[None, :]

        # Functions to be convolved
        f1 = array_to_convolve[:, None]
        g1 = Ptolemy.Gaussian_kernel(X_diff, Gaussian_sigma, args)

        # Convolution by integration
        array_convolved = jsp.integrate.trapezoid(f1 * g1, x=X_range, axis=0)
        
        # # Find peak values
        # peak_original = jnp.max(array_to_convolve)
        # peak_convolved = jnp.max(array_convolved)

        # # Adjust, i.e. normalize, convolved spectrum based on peak values
        # adjustment_factor = peak_original / peak_convolved
        # array_convolved_normalized = array_convolved * adjustment_factor
        

        # # Find areas under curves
        # area_original = jsp.integrate.trapezoid(array_to_convolve, x=X_range)
        # area_convolved = jsp.integrate.trapezoid(array_convolved, x=X_range)

        # # # Adjust, i.e. normalize, convolved spectrum based on areas
        # adjustment_factor = area_original / area_convolved
        # array_convolved_normalized = array_convolved * adjustment_factor

        array_convolved_normalized = array_convolved

        return array_convolved_normalized


    @staticmethod
    @jax.jit
    def fourier_convolution(signal, kernel):
        """
        Convolve a signal with a kernel via Fourier Transforms.
        Assumes kernel is already prepared (centered and optionally padded).
        """
        # Shift the kernel to align the zero-frequency component (representing
        # the average) with the first array element
        shifted_kernel = jnp.fft.ifftshift(kernel)

        # indices = jnp.arange(shifted_kernel.size)
        # middle_index = shifted_kernel.size // 2
        # shifted_kernel = jnp.where(indices >= middle_index, 0.0, shifted_kernel)

        # Fourier Transform of the signal and the shifted kernel
        signal_fft = jnp.fft.fft(signal)
        kernel_fft = jnp.fft.fft(shifted_kernel, n=len(signal))

        # Convolution in the frequency domain
        convolved_fft = signal_fft * kernel_fft

        # Inverse Fourier Transform
        convolved_signal = jnp.fft.ifft(convolved_fft)

        # Taking the real part of the result and normalize
        normalized_signal = jnp.real(convolved_signal) / jnp.sum(kernel)

        return normalized_signal



    @staticmethod
    @jax.jit
    def ZhangZhang2018_overdensity(m_nu):
        return 76.5 * m_nu**2.21