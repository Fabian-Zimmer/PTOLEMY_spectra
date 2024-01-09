from Shared.shared import *

#
# Determine length of x-axis range based on resolution of Fermi-Dirac spectra
#

# Neutrino parameters
m_lightest = 300*Params.meV
ordering = 'NO'
nu_masses = Physics.neutrino_masses(m_lightest, ordering, Params())
nature = 1.

# Momentum sampling (resolution) parameters for Fermi-Dirac spectra
p_start, p_stop, p_num = 0.01*Params.T_CNB, 400*Params.T_CNB, 200

# Energies and spectra for neutrino masses
E_FD, spec_FD = Ptolemy.dGamma_CNB_dE_e_FD(
    nu_masses, ordering, nature, p_start, p_stop, p_num, Params())

# Gaussian Process and convolution (i.e. fft) routine works better with smaller numbers
Energy_x = E_FD[0,:]/Params.meV
Energy_y = spec_FD[0,:]*(Params.eV*Params.yr)

# Determine energy step needed to resolve peak well enough
E_step, *_ = Ptolemy.analyze_peak_for_energy_step(
    Energy_x, Energy_y, points_in_FWHM=10)

# Construct linearly spaced energy range for Fermi-Dirac energy range
E_min, E_max = Energy_x.min(), Energy_x.max()
X_linear = jnp.arange(E_min, E_max, E_step)

# Interpolate in this energy range
Y_linear, _ = Ptolemy.gpr_interpolation(Energy_x, Energy_y, X_linear)


def plot_interpolation(x_original, y_original, x_interpolated, y_interpolated, sigma=None):

    plt.figure(figsize=(10, 6))
    plt.plot(x_original, y_original, 'r.', markersize=10, label='Original Data')
    plt.plot(x_interpolated, y_interpolated, 'b-', label='Interpolated Curve')
    # plt.fill_between(x_interpolated, y_interpolated - sigma, y_interpolated + sigma, alpha=0.2, color='b')
    plt.title('Original vs Interpolated Data using GPR')
    # plt.xlim(-12, 100)
    plt.xlim(300,300.002)
    plt.legend()
    plt.savefig('plot_interpolation.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

# plot_interpolation(Energy_x, Energy_y, X_linear, Y_linear)


# Construct total energy range with same energy step
X_min, X_max = -100, 500
X_range = jnp.arange(X_min, X_max, E_step)

# Pad interpolated array with zeros until it reaches total energy range
zeros_before = int(jnp.abs(X_min - E_min) / E_step) + 1
zeros_after  = int(jnp.abs(E_max - X_max) / E_step)

# Create padded spectrum
Y_range = jnp.concatenate((
    jnp.zeros(zeros_before),
    Y_linear,
    jnp.zeros(zeros_after)
))

# Create centered Gaussian.
Delta = 10
sigma = Delta/jnp.sqrt(8*jnp.log(2))
mu = (X_min + X_max) / 2.
Gaussian_kernel = Ptolemy.Gaussian_kernel(X_range, mu, sigma, Params())

# Free up memory
del X_linear, Y_linear, E_FD, spec_FD, Energy_x, Energy_y

# Convolution of Gaussian and spectrum with FFT method
convolved_array = Ptolemy.fourier_convolution(Y_range, Gaussian_kernel)

# print(convolved_array.shape)

def plot_convolved_spectrum(x_axis, y_axis):

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, y_axis, 'b-', label='Convolved Spectra')
    plt.title('Spectra Convolution via FFT method')
    plt.legend()
    plt.savefig('plot_convolved_spectrum.pdf', bbox_inches='tight')
    plt.close()

plot_convolved_spectrum(X_range, convolved_array)