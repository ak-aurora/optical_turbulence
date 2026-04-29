## Parameter Functions Math

The mathematical equations for each function in the library (optical_turbulence.parameters). The definition of the parameters defining the link are the following:

- $L$: total link distance between OGS and satellite.
- $z$: a point in the link distance using the on-axis link as reference.
- $h$: altitude above sea level.
- $\lambda$: wavelength
- $k=\frac{2\pi}{\lambda}$: wavenumber
- $\zeta$: zenith angle
- $C^2_{n}(h)$: refractive index structure parameter at an altitude $h$.

The order of the arguments is:
- function-specific parameters
- beam parameters
- ris model
- link information (altitude, link)

> [!info] About arrays
> `link_array` always goes from 0 to L.
> `altitude_array` has to have values in between `lct_altitude` and `sat_altitude`.
### `scint_index_DL_PR_weak`

In case of weak turbulence, scintillation index for a plane wave (downlink) can be written as

$$\sigma_{I}^{2} = \sigma_{R}^{2} = 2.25 k^{7/6} \int_{0}^{L} C_{n}^{2}(h(z)) z^{5/6}\, dz$$

### `scint_index_DL_PR_general`

$$\sigma_I^2 = \exp \left[ \frac{0.49 \sigma_R^2}{\left(1 + 1.11 \sigma_R^{12/5}\right)^{7/6}} + \frac{0.51 \sigma_R^2}{\left(1 + 0.69 \sigma_R^{12/5}\right)^{5/6}} \right]- 1$$

with $0 \leq \sigma_R^2 < \infty$


### `scint_index_DL_AA_general`
$$\sigma_I^2\left(D_G\right)= 8.70 k^{7 / 6}\times \mathrm{Re} \left\lbrace\int_{0}^L C_n^2(h(z))\left[\left(\frac{k D_G^2}{16}+i z\right)^{5 / 6}-\left(\frac{k D_G^2}{16}\right)^{5 / 6}\right]\, d z\right\rbrace$$

where $D_G$ is the "hard aperture" diameter of the receiver lens. It is important to note that aperture averaging occurs when the aperture of the receiver is sufficiently large, i.e. larger than the transverse irradiance correlation width.

### `fried_parameter_DL`

$$r_0 = \left[0.42k^2\int_{0}^R C_n^2(h(z))dz\right]^{-3/5}$$

### `isonoplanatic_angle_UL`

$$\theta_0 = \left\lbrace L\left[ 2.91 k^2  (\mu_{1u} + 0.62\mu_{2u}\Lambda^{11/6}) \right]^{3/5}\right\rbrace^{-1}$$

with

$$
\begin{align}
\mu_{1u} = &\int_{0}^{L} C_n^2(h(z)) \left[\Theta + \bar{\Theta} \left( \frac{z}{L} \right) \right] ^{5/3}\, dz\\
\mu_{2u} = &\int_{0}^L C_n^2(h(z)) \left( 1 - \frac{z}{L} \right) ^{5/3}\, dz
\end{align}
$$

### `fried_parameter_UL_TX`

The Fried parameter for a spherical wave seen from the transmitter is

$$r_{0T} = \left[0.42k^2\int_{0}^{L}C_n^2(h(z))\left(1-\frac{z}{L}\right)^{5/3}\,dz\right]^{-3/5}$$

### `fried_parameter_UL_RX`

The Fried parameter for a spherical wave seen from the receiver is

$$r_{0R} = \left[0.42k^2\int_{0}^{L}C_n^2(h(z))\left(\frac{z}{L}\right)^{5/3}\,dz\right]^{-3/5}$$

### `rytov_variance_UL_spherical `

$$\sigma_{B u}^2 =  2.25 k^{7 / 6} \int_{0}^L C_n^2(h(z))z^{5/6}\left(1-\frac{z}{L}\right)^{5/6}\,dz$$

### `scint_index_UL_spherical`

$$\sigma_{I,sph}^{2} = \exp \left\lbrace\frac{0.49 \sigma_{B u}^2}{\left[1+ 0.56 \sigma_{B u}^{12 / 5}\right]^{7 / 6}}+\frac{0.51 \sigma_{B u}^2}{\left(1+0.69 \sigma_{B u}^{12 / 5}\right)^{5 / 6}}\right\rbrace-1$$

### `rytov_variance_UL_gaussian`

$$\sigma_{B u}^2 =  8.70 k^{7 / 6}L^{5 / 6} \times \mathrm{Re} \left\lbrace\int_{0}^L C_n^2(h(z))\left[\xi^{5 / 6}[\Lambda \xi+i(1-\bar{\Theta} \xi)]^{5 / 6}-\Lambda^{5 / 6} \xi^{5 / 3}\right]\, dz\right\rbrace$$

where $$\xi= 1-z/L$$

### `scint_index_UL_gaussian`


$$\sigma_{I,sph}^{2} = \exp \left\lbrace\frac{0.49 \sigma_{B u}^2}{\left[1+ 0.56 \sigma_{B u}^{12 / 5}\right]^{7 / 6}}+\frac{0.51 \sigma_{B u}^2}{\left(1+0.69 \sigma_{B u}^{12 / 5}\right)^{5 / 6}}\right\rbrace-1$$

### `_total_beam_wander_variance_UL_gaussian`

$$\left\langle r_c^2\right\rangle= \frac{7.25L^2}{W_0^{1 / 3}} \times \int_{0}^L C_n^2(h(z))\left(1-\frac{z}{L}\right)^2\left|1-\frac{z}{F_0}\right|^{-1 / 3} dz$$

### `_bw_pointing_error_var_UL_gaussian`

$$ \sigma_{p e}^2=\left\langle r_c^2\right\rangle\left[1-\left(\frac{\pi^2 W_0^2/ r_{0 T}^2}{1+\pi^2 W_0^2 / r_{0 T}^2}\right)^{1 / 6}\right] $$

### `_bw_pointing_error_var_UL_gaussian_TT`

$$\sigma_{p e}^2=\left(\sqrt{\left\langle r_c^2\right\rangle} - T_zL\right)^2\left[1-\left(\frac{\pi^2 W_0^2/ r_{0 T}^2}{1+\pi^2 W_0^2 / r_{0 T}^2}\right)^{1 / 6}\right]$$

where

$$T_z = 0.57\left(\frac{\lambda}{2W_0}\right)\left(\frac{2W_0}{r_{0T}}\right)^{5/6}$$

### `scint_index_UL_untracked_gaussian`

$$\begin{aligned}
\sigma_{I, l}^2(0, L)_{\text {untracked }}&=34.29\left(\frac{\Lambda L}{k r_{0T}^2}\right)^{5 / 6}\left(\frac{\sigma_{p e}^2}{W^2}\right)\\&+ \exp \left\lbrace\frac{0.49 \sigma_{B u}^2}{\left[1+(1+\Theta) 0.56 \sigma_{B u}^{12 / 5}\right]^{7 / 6}}+\frac{0.51 \sigma_{B u}^2}{\left(1+0.69 \sigma_{B u}^{12 / 5}\right)^{5 / 6}}\right\rbrace-1 
\end{aligned}$$

### `scint_index_UL_tracked_gaussian`


$$\begin{aligned}
\sigma_{I, l}^2(0, L)_{\text {tracked,TC }}&=34.29\left(\frac{\Lambda_{LT} L}{k r_{0T}^2}\right)^{5 / 6}\left(\frac{\sigma_{p e,TC}^2}{W_{LT}^2}\right)\\&+ \exp \left\lbrace\frac{0.49 \sigma_{B u}^2}{\left[1+(1+\Theta) 0.56 \sigma_{B u}^{12 / 5}\right]^{7 / 6}}+\frac{0.51 \sigma_{B u}^2}{\left(1+0.69 \sigma_{B u}^{12 / 5}\right)^{5 / 6}}\right\rbrace-1 
\end{aligned}$$

where 

$$W_{LT} = W\left[1 + \left(\frac{2\sqrt{2}W_0}{r_{0T}}\right)^{5/3}\right]^{3/5}$$  (valid only for $H\gg 20$ km )

$$\Lambda_{LT} = \frac{2L}{kW_{LT}^2}$$