
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import json




#### define functions: 

def set_plot_defaults():
    """
    assign default plot settings
    """
    ## FIGURE
    #plt.rcParams["text.usetex"] = "True"

    ## FONT
    #plt.rcParams['font.serif']=['cm']
    #plt.rcParams['font.family']='serif'
    # plt.rcParams['font.serif']=['cm']
    plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

    plt.rcParams['font.size']=18
    plt.rcParams['xtick.labelsize']=16
    plt.rcParams['ytick.labelsize']=16
    plt.rcParams['legend.fontsize']=16
    plt.rcParams['axes.titlesize']=16
    plt.rcParams['axes.labelsize']=16
    plt.rcParams['axes.linewidth']=1.5
    plt.rcParams["lines.linewidth"] = 1.5
    ## TICKS
    plt.rcParams['xtick.top']='True'
    plt.rcParams['xtick.bottom']='True'
    plt.rcParams['xtick.minor.visible']='True'
    plt.rcParams['xtick.direction']='out'
    plt.rcParams['ytick.left']='True'
    plt.rcParams['ytick.right']='True'
    plt.rcParams['ytick.minor.visible']='True'
    plt.rcParams['ytick.direction']='out'
    plt.rcParams['xtick.major.width']=1.5
    plt.rcParams['xtick.minor.width']=1
    plt.rcParams['xtick.major.size']=4
    plt.rcParams['xtick.minor.size']=3
    plt.rcParams['ytick.major.width']=1.5
    plt.rcParams['ytick.minor.width']=1
    plt.rcParams['ytick.major.size']=4
    plt.rcParams['ytick.minor.size']=3

def count_lines(wave, flux, mask, prominence=0.00001):
    """
    Find significant absorption lines.
    prominence : minimum line depth relative to local continuum.
    """
    w = wave[mask]
    f = flux[mask]
    f_norm = f / np.nanmax(f) # Normalize so median continuum is ~1
    f_smooth = gaussian_filter1d(f_norm, sigma=1) # Light smoothing to avoid counting tiny numerical bumps
    peaks, properties = find_peaks(-f_smooth,prominence=prominence,distance=3) # Absorption lines are peaks in the inverted spectrum
    return len(peaks), w[peaks], f[peaks]


# Place a two-column file (wavelength in nm, transmission 0..1) next to
# this script. Set this name to match your local file.
TELLURIC_FILE = Path(__file__).resolve().parent / "tellurics_500k.csv"
TELLURIC_INPUT_R = 500000 #500k telluric input gives plenty of headroom


def load_telluric_spectrum(filename=TELLURIC_FILE):
    """Load telluric spectrum, in nm"""
    path = Path(filename)
    arr = np.loadtxt(path, comments="#", delimiter="," if path.suffix.lower() == ".csv" else None, usecols=(0, 1), encoding="utf-8-sig")
    wave, transmission = arr[:, 0], arr[:, 1]
    return wave, transmission


def convolve_telluric(wave_nm, transmission, target_R,
                      input_R=TELLURIC_INPUT_R):
    """convolve spectrum to chosen resolution. NOTE: this is done in the telluric sampling section
    """
    if not (0 < target_R <= input_R):
        raise ValueError("Require 0 < target_R <= input_R")
    ln_wave = np.log(wave_nm)
    input_step = np.min(np.diff(ln_wave))
    desired_step = 1.0 / (3.0 * target_R)
    step = min(input_step, desired_step)
    n = int(np.ceil((ln_wave[-1] - ln_wave[0]) / step)) + 1
    log_grid = np.linspace(ln_wave[0], ln_wave[-1], n)
    step = log_grid[1] - log_grid[0]
    sampled = np.interp(log_grid, ln_wave, transmission)
    fwhm_log = np.sqrt(target_R**-2 - input_R**-2)
    sigma_pixels = fwhm_log / (2 * np.sqrt(2 * np.log(2)) * step)
    if sigma_pixels > 0:
        sampled = gaussian_filter1d(sampled, sigma_pixels, mode="nearest", truncate=5.0)
    return np.exp(log_grid), np.clip(sampled, 0, 1)


def _make_telluric_mask(bands, transmission_threshold=0.6):
    wave, transmission = load_telluric_spectrum()
    wave_parts, trans_parts = [], []

    samples_per_res = 3

    for low, high, resolution in bands:

        # Select wavelength range with margin for convolution
        margin = 6.0 * high / resolution
        select = (wave >= low - margin) & (wave <= high + margin)

        if np.count_nonzero(select) < 3:
            raise ValueError(f"Insufficient telluric coverage for {low}-{high} nm")

        # Convolve to the instrument resolution
        w, t = convolve_telluric(
            wave[select], transmission[select], resolution
        )

        # Create logarithmic wavelength grid with 3 samples per resolution element
        dln_wave = 1.0 / (resolution * samples_per_res)
        new_wave = low * np.exp(
            np.arange(int(np.ceil(np.log(high / low) / dln_wave))) * dln_wave
        )
        new_wave = new_wave[new_wave < high]

        # Interpolate convolved spectrum onto new wavelength grid
        new_transmission = np.interp(new_wave, w, t)

        wave_parts.append(new_wave)
        trans_parts.append(new_transmission)

    # Combine wavelength bands
    wave_nm = np.concatenate(wave_parts)
    transmission = np.concatenate(trans_parts)

    # Apply telluric transmission threshold
    telluric_mask = transmission > transmission_threshold
    good_waves = wave_nm[telluric_mask]

    return wave_nm, transmission, telluric_mask, good_waves


def make_telluric_mask_MICH(transmission_threshold=0.6, res_L= 50000, res_M = 100000):
    return _make_telluric_mask([(2800, 4200, res_L),
                                (4500, 5500, res_M)],
                               transmission_threshold)


def make_telluric_mask_CRIRES(transmission_threshold=0.6):
    return _make_telluric_mask([(2800, 4200, 92000),
                                (4500, 5500, 92000)],
                               transmission_threshold)

def make_crires_mask(wave_nm, telluric_mask, mol):
    """
    Build the CRIRES+ wavelength mask and combine  with the telluric transmission mask.
    """
    ranges_M4368 = [
        (3512.841, 3582.734),
        (3747.081, 3821.617),
        (4014.774, 4094.586),
        (4323.720, 4409.760),
        (4683.789, 4776.814),
        (5109.398, 5210.712)
    ]

    ranges_L3340 = [
        (2810.364, 2866.354),
        (2958.407, 3017.240),
        (3122.778, 3184.854),
        (3306.456, 3372.157),
        (3513.062, 3582.824),
        (3747.144, 3821.502),
        (4014.595, 4094.191)
    ]

    if mol in ['12C-1H4', '1H2-32S', '14N-1H3',  '1H2-16O']:
        ranges = ranges_L3340
    else: 
        ranges = ranges_M4368


    crires_mask = np.zeros(wave_nm.shape, dtype=bool)
    for wmin, wmax in ranges: crires_mask |= (wave_nm >= wmin) & (wave_nm <= wmax)
    combined_mask_crires = crires_mask & telluric_mask
    return combined_mask_crires


def make_michelangelo_mask(wave_nm, telluric_mask):
    """
    Build the michelangelowavelength mask and combine with the telluric transmission mask.
    """

    mich_mask = (wave_nm >= 2900) & (wave_nm <= 5500)
    combined_mask_mich = mich_mask & telluric_mask

    return combined_mask_mich


######## Main:
set_plot_defaults()
fig, ax = plt.subplots(figsize= (10,8))
############## to edeit...
# Telluric masks are independent of planetary model / observing mode.

######## JULIA: this is where to change the MIChelangelo target resolution
RES_L_mich = 65000 ### play around with these to see how reolution impacts
RES_M_mich = 130000 ####

# Michelangelo wavelength ranges (nm)
WAVE_L_MIN, WAVE_L_MAX = 2800, 4200
WAVE_M_MIN, WAVE_M_MAX = 4500, 5500

# Build a descriptive tag for output file names
mich_tag = f"L{WAVE_L_MIN}-{WAVE_L_MAX}nm_R{RES_L_mich}_M{WAVE_M_MIN}-{WAVE_M_MAX}nm_R{RES_M_mich}"

wave_nm_MICH, transmission_MICH, telluric_mask_MICH, good_waves_MICH = make_telluric_mask_MICH(res_L = RES_L_mich, res_M = RES_M_mich)
wave_nm_CRIRES, transmission_CRIRES, telluric_mask_CRIRES, good_waves_CRIRES = make_telluric_mask_CRIRES()
combined_mask_mich = make_michelangelo_mask(wave_nm_MICH, telluric_mask_MICH)

modes = ['trans', 'sp_em', 'DI_em']

# Dict to collect all results for saving to file
all_results = {
    "michelangelo_params": {
        "L_band_range_nm": [WAVE_L_MIN, WAVE_L_MAX],
        "L_band_resolution": RES_L_mich,
        "M_band_range_nm": [WAVE_M_MIN, WAVE_M_MAX],
        "M_band_resolution": RES_M_mich,
    },
    "molecules": ['12C-1H4', '1H2-32S', '28Si-16O', '14N-1H3', '12C-16O', '1H2-16O', '12C-16O2'],
    "modes": {}
}

for mode in modes:

    if mode == 'trans':
        obs_type = 'short-period' # other option:'directly-imaged' -this just changes how we handle the contrast improvement from better AO (assumed theat MICHELANGELO is behind a DSM)
        model_array = np.array(['spectra_600K_iso_low_mu_Jupiter.npz', 'spectra_1100K_iso_low_mu_Jupiter.npz', 'spectra_1700K_iso_low_mu_Jupiter.npz', 'spectra_2200K_iso_low_mu_Jupiter.npz'])
        model_names = np.char.add('Transmission ', np.array(['600 K low MMW', '1100 K low MMW','1700 K low MMW','2200 K low MMW']))
        plot_array = np.array([0,1,2,3])

    if mode == 'sp_em':
        obs_type = 'short-period' # other option:'directly-imaged' -this just changes how we handle the contrast improvement from better AO (assumed theat MICHELANGELO is behind a DSM)
        model_array = np.array(['emission_600K_guillot_low_mu_Jupiter.npz', 'emission_1100K_guillot_low_mu_Jupiter.npz', 'emission_1700K_guillot_low_mu_Jupiter.npz', 'emission_2200K_guillot_low_mu_Jupiter.npz'])
        model_names = np.char.add('Emission ', np.array(['600 K low MMW', '1100 K low MMW','1700 K low MMW','2200 K low MMW']))
        plot_array = np.array([4,5,6,7])

    if mode == 'DI_em':
        obs_type = 'directly-imaged' # other option:'directly-imaged' -this just changes how we handle the contrast improvement from better AO (assumed theat MICHELANGELO is behind a DSM)
        model_array = np.array(['emission_600K_guillot_low_mu_Jupiter.npz', 'emission_1100K_guillot_low_mu_Jupiter.npz', 'emission_1700K_guillot_low_mu_Jupiter.npz', 'emission_2200K_guillot_low_mu_Jupiter.npz'])
        model_names = np.char.add('Direct Imaging ', np.array(['600 K low MMW', '1100 K low MMW','1700 K low MMW','2200 K low MMW']))
        plot_array = np.array([8,9,10,11])



    ###############

    snr_improv_array = np.ones((len(model_array), 7))

    #generate telluric mask for michelangelo and CRIRES+ _----> NOTE: This includes the resolution!!

    for i, (model, name) in enumerate(zip(model_array, model_names)):

        ################ import spectra ################ to edit...
        filename = f'models/JS_models/{model}'
        data = np.load(filename)

        mols = ['12C-1H4', '1H2-32S', '28Si-16O', '14N-1H3', '12C-16O', '1H2-16O', '12C-16O2']
        for j, mol in enumerate(mols):

            combined_mask_crires = make_crires_mask(wave_nm_CRIRES, telluric_mask_CRIRES, mol) #has to be made here as mol is needed to choose right comparison CRIRES grating

            spec_planet = data[mol][1]
            wave_planet = data[mol][0]*1e7

            #interp the model onto telluric grid, which has the right R and lambda range already.
            if mode == 'trans': fact =-1 #transmission liens are 'upside down'
            else: fact=1
            interp_spec_planet_CRIRES = np.interp(wave_nm_CRIRES, wave_planet, spec_planet)*fact # these are different to handle the fact that the resolutions between the two can be different
            interp_spec_planet_MICH = np.interp(wave_nm_MICH, wave_planet, spec_planet)*fact #the resolution itself is set within the interp model. This is just how I set it up, I appreciate this is not very user friendly, mais c'est la vie

            #work out N_lines in each masked model. The actual numbers here are less important as it is mostly just the relative number between crires and Mic that dictates the S/N scaling
            N_lines_crires, line_wave_crires, line_flux_crires = count_lines(wave_nm_CRIRES, interp_spec_planet_CRIRES, combined_mask_crires)
            N_lines_mich, line_wave_mich, line_flux_mich = count_lines(wave_nm_MICH, interp_spec_planet_MICH, combined_mask_mich)

            print(N_lines_crires)
            print(N_lines_mich)

            N_lines_fact = N_lines_mich/N_lines_crires
            rel_blaze_eff = 1.49 #average from the Mich ETC, improvement over CRIRES
            res_SNR_gain = np.sqrt(92000/((RES_L_mich + RES_M_mich)/2)) #vs R=92,000 CRIRES. can turn off if needed.
            throughput_fact = res_SNR_gain* rel_blaze_eff




            ### if we want to use the ERIS PSF...
            if obs_type == 'directly-imaged': contrast_fact = 2e-3/8e-5 #improvement in raw contrast from ERIS PSF in L-prime vs CRIRES+ at 3.8 microns (both for beta pic in 3h on-sky exposures). have to make an assumption on seperation, so 1 arcsec chosen.
            elif obs_type =='short-period': contrast_fact = 2 #from the improvement in strehl with the DSM PSF vs the CRIRES MACAO (again measured for beta pic for both instruments). For ~90% encerciled energy, extracted noise for ERIS PSF is approx 2 times less

            print(f'MICH vs CRIRES throughput: {throughput_fact}')                                                                                                                                        
            print(f'MICH vs CRIRES wavelength coverage: {N_lines_fact}')
            print(f'MICH vs CRIRES contrast: {contrast_fact}')
            
            snr_improv = np.sqrt(N_lines_fact) *np.sqrt(throughput_fact) * np.sqrt(contrast_fact) #S_to_noise scaling vs crires for michelangelo, for this particulat model
            print(f'S/N factor improvement for Michelangelo vs CRIRES+: {snr_improv} for {name} mode {mol} molecule') #for context, the S/N factor improvement for METIS is ~8, for similarly hand-wavey scalings. MICH would have ~20x the wavelength coverage of METIS, which is the main drive here.  

            snr_improv_array[i, j] = snr_improv #save the value


    print(snr_improv_array)

    # Store results for this mode
    all_results["modes"][mode] = {
        "model_names": list(model_names),
        "snr_improvement": snr_improv_array.tolist()   # shape: (n_models, n_mols)
    }

    mols = ['12C-1H4', '1H2-32S', '28Si-16O', '14N-1H3', '12C-16O', '1H2-16O', '12C-16O2']
    mol_names = ['CH$_4$', 'H$_2$S', 'SiO', 'NH$_3$', 'CO', 'H$_2$O', 'CO$_2$']
    cols = ['teal', 'purple', 'darkorange', 'crimson', 'blue', 'cyan', 'magenta']
    for j, mol in enumerate(mols):
        if mode == 'trans': plt.errorbar(plot_array, snr_improv_array[:,j], label = mol_names[j], color = cols[j], marker = 'o')
        else: plt.errorbar(plot_array, snr_improv_array[:,j], color = cols[j], marker = 'o')


plt.fill_between([-0.5, 3.5], -0.1, 100, color= 'darkorange', alpha = 0.2)
plt.fill_between([3.5, 7.5], -0.1, 100,color= 'crimson', alpha = 0.2)
plt.fill_between([7.5, 11.5], -0.1, 100,color= 'purple', alpha = 0.2)

#plt.yscale('log')
plt.ylim(0, 18)
plt.xlabel('Models')
plt.ylabel('S/N improvement on CRIRES+')

all_model_names = ['600 K', '1100 K','1700 K','2200 K', '600 K', '1100 K','1700 K','2200 K', '600 K', '1100 K','1700 K','2200 K']
ax.set_xticks(np.arange(12))
ax.set_xticklabels(all_model_names, rotation=45, ha='right')

plt.text(0, 1, 'Transmission', fontsize = 18)
plt.text(4.5, 1, 'Emission', fontsize = 18)
plt.text(8, 1, 'Directly Imaged', fontsize = 18)

plt.xlim(-0.5, 11.5)
plt.legend()

# Save plot and data with descriptive filenames
plot_filename = f'SNR_comparison_plot_{mich_tag}.png'
data_filename = f'SNR_comparison_data_{mich_tag}.txt'

plt.savefig(plot_filename, dpi=300)
plt.show()

# Save results to a JSON-formatted .txt file (load back with json.load or json.loads)
with open(data_filename, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nPlot saved to: {plot_filename}")
print(f"Data saved to: {data_filename}")
print("Data structure:")
print("  all_results['michelangelo_params']  -> instrument settings")
print("  all_results['molecules']             -> list of molecule keys")
print("  all_results['modes']['trans']        -> transmission mode results")
print("  all_results['modes']['sp_em']        -> short-period emission results")
print("  all_results['modes']['DI_em']        -> directly-imaged emission results")
print("  each mode has 'model_names' and 'snr_improvement' (shape: n_models x n_molecules)")
