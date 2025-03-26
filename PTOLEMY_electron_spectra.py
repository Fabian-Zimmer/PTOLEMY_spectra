from Shared.shared import *
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning from sklearn.gaussian_process.kernels
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.gaussian_process.kernels')


def PLOT_Ptolemy_electron_spectra(spectra_list, m_lightest_arr, Delta_arr):

    # Define the set of valid spectra options
    valid_spectra = [
        "tritium_bkg_true", 
        "tritium_bkg_smeared",
        "fermi_dirac_true",
        "fermi_dirac_smeared",
        "total_true",
        "total_smeared",
        "gaussian_kernel",
        "CNB_sim_smeared"]

    # Helper function to get a string of valid options
    def get_valid_options_str():
        return f"Valid options: {', '.join(valid_spectra)}"

    # Check if spectra_list is empty or contains invalid items
    if not spectra_list:
        raise ValueError("spectra_list is empty. " + get_valid_options_str())
    elif any(spectrum not in valid_spectra for spectrum in spectra_list):
        raise ValueError("Invalid input in spectra_list. " + get_valid_options_str())


    # Lighest neutrino masses to anchor hierarchy
    # m_lightest_arr = jnp.array([10, 50, 100, 300])*Params.meV  # original

    # PTOLEMY experiment energy resolutions [meV]
    # Delta_arr = [20, 20, 20]  # original

    # Explore both mass orderings
    orders = [
        'NO', 
        'IO'
    ]
    colors = ['red', 'blue']
    sim_colors = ['magenta', 'green']

    # Neutrino nature
    nature = 1.  # 1 = Dirac neutrinos, 2 = Majorana neutrinos

    # Momentum sampling (resolution) parameters for Fermi-Dirac spectra
    p_start = Params.p_start * Params.T_CNB
    p_stop  = Params.p_stop  * Params.T_CNB
    p_num   = Params.p_num

    # Unit for y-axis, i.e. rates per year per energy
    y_unit = Params.yr*Params.eV

    # Plot 4 combinations for the parameters of the PTOLEMY experiment setup
    plt.figure(figsize=(12,10))
    for i in range(4):
        ax = plt.subplot(221 + i)

        # if i != 0:  # for testing
        #     continue

        m_lightest = m_lightest_arr[i]
        Delta = Delta_arr[i]

        # Standard deviation for Gaussian convolution kernel
        sigma = Delta/jnp.sqrt(8*jnp.log(2))

        for ordering, color, sim_color in zip(orders, colors, sim_colors):

            # Get neutrino masses
            nu_masses = Physics.neutrino_masses(m_lightest, ordering, Params())

            # Energies and spectra for neutrino masses
            Es_FD, specs_FD = Ptolemy.dGamma_CNB_dE_e_FD(
                nu_masses, ordering, nature, p_start, p_stop, p_num, Params())
            
            # Work with smaller numbers for numerical stability
            Es_FD /= Params.meV
            specs_FD *= (Params.eV*Params.yr)

            # Determine energy step needed to resolve smallest peak well enough
            # (this is the one for the highest mass)
            E_step, *_ = Ptolemy.analyze_peak_for_energy_step(
                Es_FD[-1,:], specs_FD[-1,:], points_in_FWHM=10)

            # Construct total energy range with same energy step
            # Construct total x-axis (energy) range for spectrum with same energy step
            # (similar to PTOLEMY et al. (2019) Fig. 1, labeled as: E_e - E_{end,0} [meV])
            X_min, X_max = -160, 350
            X_range = jnp.arange(X_min, X_max, E_step)

            # Construct centered Gaussian kernel
            mu = (X_min + X_max) / 2.
            Gaussian_kernel = Ptolemy.Gaussian_kernel(X_range, mu, sigma, Params())
            # Gaussian_kernel = jnp.where(Gaussian_kernel <= 1e-2, 0.0, Gaussian_kernel)

            # Convert to energy range
            E_range = X_range*Params.meV + Params.m_e + Params.K0_end

            if ordering == "NO" and "gaussian_kernel" in spectra_list:
                ax.semilogy(
                    X_range-50, Gaussian_kernel,
                    color='orange', ls='dashdot', alpha=0.5, label=f'Gaussian') 

                # Check if Gaussian is normalized
                # Gaussian_area = jsp.integrate.trapezoid(Gaussian_kernel, x=X_range)
                # ic(Gaussian_area)


            # =============== #
            # Tritium Spectra #
            # =============== #

            # Tritium beta-decay (background) spectrum
            T_beta_spec_true = Ptolemy.Beta_decay_spectrum(
                E_range, nu_masses, ordering, Params())

            # Convolved (i.e. smeared) Trititum beta-decay spectrum
            T_beta_spec_conv = Ptolemy.fourier_convolution(T_beta_spec_true*y_unit, Gaussian_kernel)

            
            # Fourier transform method assumes periodic signal, which beta-spectra are not
            # Therefore, we manually remove artifact on the right side of the plot
            physical_range = (X_range < 100)
            T_beta_spec_conv *= physical_range


            # =================== #
            # Fermi-Dirac Spectra #
            # =================== #

            if "fermi_dirac_true" or "fermi_dirac_smeared" or "total_smeared" in spectra_list:

                spectra_FD_true_l = []
                for m_nu, E_FD, spec_FD in zip(nu_masses, Es_FD, specs_FD):

                    # Gaussian Process and convolution (i.e. fft) routine works better with smaller numbers
                    E_min, E_max = E_FD.min(), E_FD.max()

                    # Construct linearly spaced energy ranges for Fermi-Dirac energy ranges
                    X_linear = jnp.arange(E_min, E_max, E_step)

                    # Interpolate in these energy ranges
                    Y_linear, _ = Ptolemy.gpr_interpolation(E_FD, spec_FD, X_linear)
                    
                    # Pad interpolated array with zeros to match size of total energy range
                    zeros_bef = int(jnp.abs(X_min - E_min) / E_step)
                    zeros_aft = int(jnp.abs(E_max - X_max) / E_step)
                    Y_range = jnp.concatenate((
                        jnp.zeros(zeros_bef),
                        Y_linear,
                        jnp.zeros(zeros_aft)
                    ))

                    # Add additional zero(s) to perfectly match X_range
                    diff = len(X_range) - len(Y_range)
                    if diff != 0:
                        Y_range = jnp.concatenate((Y_range, jnp.zeros(diff)))

                    # Boost by gravitational clustering (analytical method)
                    boost = 1 + Ptolemy.ZhangZhang2018_overdensity(m_nu)
                    Y_range *= boost

                    spectra_FD_true_l.append(Y_range)
                
                # Sum all Fermi-Dirac spectra
                spectra_FD_true = jnp.sum(jnp.array(spectra_FD_true_l), axis=0)

                
                if "fermi_dirac_smeared" or "total_smeared" in spectra_list:

                    # Convolution via FFT method
                    spectra_FD_conv = Ptolemy.fourier_convolution(spectra_FD_true, Gaussian_kernel)
                

                if "fermi_dirac_true" in spectra_list:
                    ax.semilogy(
                        X_range, spectra_FD_true,
                        color=color, ls='solid', alpha=0.7, label=f'{ordering}')
                    

                if "fermi_dirac_smeared" in spectra_list:
                    ax.semilogy(
                        X_range, spectra_FD_conv,
                        color=color, ls='--', alpha=0.7, label=f'{ordering}')



            # ================== #
            # Simulation Spectra #
            # ================== #           

            if "CNB_sim_smeared" in spectra_list:

                Es_sim, specs_sim = Ptolemy.dGamma_CNB_dE_e_simulation(
                    'halo13', nu_masses, ordering, nature, Params())

                Es_sim /= Params.meV
                specs_sim *= (Params.eV*Params.yr)

                spectra_sim_true_l = []
                for E_sim, spec_sim in zip(Es_sim, specs_sim):

                    # Gaussian Process and convolution (i.e. fft) routine works better with smaller numbers
                    E_min, E_max = E_sim.min(), E_sim.max()

                    # Construct linearly spaced energy ranges for CNB sim energy ranges
                    X_linear = jnp.arange(E_min, E_max, E_step)

                    # Interpolate in these energy ranges
                    Y_linear, _ = Ptolemy.gpr_interpolation(E_sim, spec_sim, X_linear)
                    
                    # Pad interpolated array with zeros to match size of total energy range
                    zeros_bef = int(jnp.abs(X_min - E_min) / E_step)
                    zeros_aft = int(jnp.abs(E_max - X_max) / E_step)
                    Y_range = jnp.concatenate((
                        jnp.zeros(zeros_bef),
                        Y_linear,
                        jnp.zeros(zeros_aft)
                    ))

                    # Add additional zero(s) to perfectly match X_range
                    diff = len(X_range) - len(Y_range)
                    if diff != 0:
                        Y_range = jnp.concatenate((Y_range, jnp.zeros(diff)))

                    spectra_sim_true_l.append(Y_range)
                
                # Sum all CNB sim spectra
                spectra_sim_true = jnp.sum(jnp.array(spectra_sim_true_l), axis=0)

                # Convolution via FFT method
                spectra_sim_conv = Ptolemy.fourier_convolution(spectra_sim_true, Gaussian_kernel)
            
                ax.semilogy(
                    X_range, spectra_sim_conv, color=sim_color)
            


            # ========================= #
            # Plot Selection of Spectra #
            # ========================= #

            if "tritium_bkg_true" in spectra_list:
                ax.semilogy(
                    X_range, T_beta_spec_true*y_unit,
                    color=color, ls='dotted')
                
            if "tritium_bkg_smeared" in spectra_list:
                ax.semilogy(
                    X_range, T_beta_spec_conv,
                    color=color, ls='solid', alpha=0.7, label=f'{ordering}')

            if "total_smeared" in spectra_list:
                total_smeared = T_beta_spec_conv + spectra_FD_conv
                ax.semilogy(
                    X_range, total_smeared,
                    color=color, ls='solid', alpha=0.7, label=f'{ordering}')

            ax.text(0.95, 0.95, 
                    rf'$m_\mathrm{{lightest}}={m_lightest/Params.meV:.0f}$ meV'
                    '\n'
                    rf'$\Delta={Delta:.0f}$ meV',
                    fontsize=10, va='top', ha='right', 
                    transform=ax.transAxes)

            ax.legend(loc='center left', prop={'size': 10})
            ax.set_xlabel(r'$E_e - E_{\mathrm{end},0}$ [meV]', fontsize=16)
            ax.set_ylabel(r'$d\Gamma / dE_e$ $(yr^{-1} eV^{-1})$', fontsize=16)
            ax.set_xlim(X_min+10, X_max)
            ax.set_ylim(1e-2, 1e11)
            
    plt.suptitle('PTOLEMY electron energy spectra')
    plt.tight_layout()
    plt.savefig('PTOLEMY_electron_spectra_simulation.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


PLOT_Ptolemy_electron_spectra(
    spectra_list=(
        "tritium_bkg_true", 
        # "tritium_bkg_smeared",
        # "fermi_dirac_true",
        "fermi_dirac_smeared",
        # "total_true",
        # "total_smeared",
        # "gaussian_kernel",
        "CNB_sim_smeared",
        ),
    m_lightest_arr=jnp.array([10, 50, 100, 200])*Params.meV,
    Delta_arr=[20, 50, 50, 100]
    )