# mueller-sph with configuration for rayleigh-taylor instability

Original:
A concise 2D implementation of Müller's interactive smoothed particle hydrodynamics (SPH) paper in C++.
Please see the accompanying writeup [here](https://lucasschuermann.com/writing/implementing-sph-in-2d)

Fork:
In this version we changed constants and initialization of the particles to make the observation of rayleigh-taylor instabilities posible. Also there is a second (denser) fluid spawning above a lighter fluid. Additionally the fluids are initialized with a density according to the relation between pressure and height. So the first fluid (blue) fills the space from the bottom to VIEW_WIDTH/2 and the other one (red) the space above. The gas constant of the first fluid (GAS_CONSTANT) is chosen so the fluid is more or less homogeneous. Teh second gas constant (GAS_CONSTANT2) is calculated for an equilibrium of pressure at the interface.
Other changes include:
  BOUND_DAMPING = 0.95 to avoid having particles sticking to the borders
  changed formula to calculate the force f_press
  added exponent GAMMA to equation p = R*rho^gamma
  
As reference to the kernels used, look here (Müller, 2003):
https://matthias-research.github.io/pages/publications/sca03.pdf


## License
[MIT](https://lucasschuermann.com/license.txt)
