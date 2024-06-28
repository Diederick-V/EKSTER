import os
import numpy             as np
import astropy.constants as c
import astropy.units     as u
import matplotlib.pyplot as plt
from matplotlib.colors   import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.figure   import Figure
from matplotlib.axes     import Axes
from matplotlib          import cycler
from scipy.optimize      import fsolve
from scipy.integrate     import quad
from scipy.special       import spence, hyp2f1
from datetime            import datetime as dt
from functools           import cached_property

#      ______________________________________________       
#     |   ___||  |/  /  /  ____    ___    ___||   _  \      
#     |  |__  |  '  /  |  (    |  |   |  |__  |  |_)  |     
#     |   __| |     \  \   \   |  |   |   __| |      /      
#     |  |____|  |\  \__)   |  |  |   |  |____|  |\  \_     
#     |__________| \_______/   |__|   |__________| \___|    
# Encoding Kinematic and SIDM Theories in EoS-varied RELHICs
# By D.T.A. Vroom                           #######        
#                                        #####  ######     
#                                       ################   
#                                      ####################
#                                    ###############       
#                                 ####  ###########        
#                               #       ###########        
#                             ##     ###  #########        
#                            ##   ###     ########         
#                            #  ###        ######          
#                           ######          #####          
#                         ######            ####           
#                        #######            ##             
#                       #######             #              
#                     ##########          ###              
#                    ##########         ###                
#                     ######## ##     ####                 
#                    ###################                   
#                   ##################                     
#                  ##################        #####         
#                ##########     ##  ##     ######          
#               #########       ##   ##  #######           
#             #########          ##   ########             
#            #########           ### #######               
#           ########            ##############       ######
#         #######            ####################  ######  
#        #######            ############################   
#      #######      #################################      
#     #######  ##########################                  
#    #####  #####################                          
#   ####   #################                               
#        ###################                               
#    #########################                             
# ##############            ####                           
# ##########                  #####                        

def str_convert(x: str) -> bool | float:
    """
    Converts the `x` to `True` or `False`, if possible, and else converts it to a float.

    This function was needed for converting the value of the parameter `concentration` in the `HaloModel`.

    Parameters
    ----------
    `x`: bool or float

    Returns
    -------
    x_conv: bool or float
            `False` if `x='False'`,
            `True`  if `x='True'`,
            else float
    
    Examples
    --------
    >>> str_convert('True')
    True
    >>> str_convert('False')
    False
    >>> str_convert('42')
    42.0
    """
    if x in ["False", "True"]:
        return x == "True"
    else:
        return float(x)

# DONE 4 plots: purple plot (c=11.4), purple plot (neto), purple plot (neto + 1sigma), purple plot (neto - 1sigma)  
# DONE reduce mass range to just interesting bits (log = 8.5 )
# DONE include cloud 9 arrows + M200 range

# DONE extra crossesctions (log(sigma) < 0), more contrast between colors

class HaloModel:
    def __init__(self, 
                 concentration:                  bool | float = True,
                 M200:            u.quantity.Quantity | float = None,
                 H0:              u.quantity.Quantity | float = 70.4*u.km/u.s/u.Mpc,
                 scatteringParam: u.quantity.Quantity | float = 0*u.cm**2/u.g,
                 haloAge:         u.quantity.Quantity | float = 10*u.Gyr,
                 directory:                      bool | str   = True):
        """
        Created by D.T.A. Vroom as part of his Master Research Project to complete his masters in Astronomy & Cosmology 
        at Leiden University. This project is supervised by Matthieu Schaller.
        
        Last updated: 2024-05-08

        Class for modelling starless dark matter haloes in hydrostatic equilibrium. (RELHICs: REionization-Limited HI Clouds)\n
        Most of the calculations are based on the paper by Benitez-Llambay et al. (2017), most notably Appendix A.

        Parameters
        ----------
        `concentration`: float or bool (unitless) \n
        \t The concentration of the dark matter halo, usually defined as `c = R200/r_s`.\n
        \t With `R200` the virial radius, and `r_s` the normalization radius in the NFW-profile.\n

        \t `concentration` can either be a `float` or a `bool`. If a `bool` is given,\n
        \t `concentration` is calculated based on a M200-concentration relationship given in Neto et al. (2007).\n
        \t `True` for a relaxed system, `False` for an unrelaxed system. \n
        \t Usually a relaxed system is assumed for RELHICs.
        \n
        \n
        `M200`: float or astropy quantity (M_sun) \n
        \t The virial mass of the dark matter halo, i.e. the mass of the halo at radius `R200`. \n
        \t `M200` is the defining characteristic of a dark matter halo. \n

        \t `M200` can either be a `float` or an `astropy quantity`.\n
        \t If it is an `astropy quantity`, it has to have equivalent units of mass, usually solar masses (M_sun). \n
        \t If it is a `float`, `M200` is assumed to be in solar masses and converted to an `astropy quantity` object\n
        \t with such units.
        \n
        `H0`: float or astropy quantity (km/s/Mpc) \n
        \t The assumed Hubble constant for the calculations. Mostly used for calculating the critical density. \n

        \t `H0` can either be a `float` or an `astropy quantity`.\n
        \t If it is an `astropy quantity`, it has to have equivalent units of frequency, usually km/s/Mpc.\n
        \t If it is a `float`, `H0` is assumed to be in km/s/Mpc and converted to an `astropy quantity` object\n
        \t with such units. 
        \n
        `scatteringParam`: float or astropy quantity (cm^2/g) \n
        \t The scattering parameter used for introducing self interacting dark matter (SIDM).\n
        \t This parameter is equal to the the quantity `sigma/m`, with `sigma` the scattering cross section,\n
        \t roughly proportional to the probability of an interaction occurring \n
        \t and `m` is the mass of a dark matter particle. \n

        \t `scatteringParam` can either be a `float` or an `astropy quantity`.\n
        \t If it is an `astropy quantity`, it has to have equivalent units of an area divided by a mass,\n
        \t usually cm^2/g. \n
        \t If it is a `float`, `scatteringParam` is assumed to be in cm^2/g and converted to an \n
        \t`astropy quantity` object with such units. 
        
        \n
        `haloAge`: float or astropy quantity (Gyr) \n
        \t Age of the dark matter halo, usually assumed to be around the age of the universe (approx 10 Gyr) \n
                   
        \t `haloAge` can either be a `float` or an `astropy quantity`. \n
        \t If it is an `astropy quantity`, it has to have equivalent units of time, usually Gyr. \n
        \t If it is a `float`, `haloAge` is assumed to be in Gyr and converted to an `astropy quantity` object \n
        \t with such units. 

        Subclasses
        ----------
        This class contains several subclasses. Here is a comprehensive overview of each subclass:

        `HaloModel`:

        `NFW`:

        `EoS`:

        `densProfile`:

        `HProfile`:

        `SIDM`:

        `Notes`:

        `Plotter`:

        """
        
        self.M200Input            = M200
        self.H0Input              = H0 
        self.conInput             = concentration
        self.scatteringParamInput = scatteringParam
        self.haloAgeInput         = haloAge

        self.directory = directory

        self.NFW         = NFW(self)
        self.EoS         = EoS(self)
        self.densProfile = densProfiles(self)
        self.HProfile    = Hydrogen_Profile(self)
        self.SIDM        = selfInteractingDarkMatter(self)
        self.plot        = Plotter(self)

        if M200 is None and self.directory != False:
            self.load(directory)

        if self.directory or (type(self.directory) is str):
            self.timeOfCreation = dt.now()
            self.folder         = self.timeOfCreation.strftime("%Y-%m-%d_%H%M%S")

            if type(self.directory) is str: self.folder = self.directory

            self.path     = os.getcwd() + "/Simulations/"
            self.fullPath = self.path + self.folder + "/"

            if self.directory: os.makedirs(self.fullPath, exist_ok=True)
            
            self.notes = Notes(self)
        
    @property
    def M200(self) -> u.quantity.Quantity:
        """
        Virial dark matter mass of the halo. \n
        The mass contained within a radius R200.\n
        Usual unit: Solar Masses
        """
        if type(self.M200Input) is not u.quantity.Quantity:
            return self.M200Input * u.M_sun
        else:
            return self.M200Input
        
    @M200.setter
    def M200(self, __new):
        self.M200Input = __new
        #Need to delete cached values of all variables which depend on M200 when M200 changes
        for attr in ('R200', 'rs', 'V200', 'T200', 'transRadius', 'transDens', 'transMass', 'KingRadius', 'centralDens'):
            self.__dict__.pop(attr, None)
            self.NFW.__dict__.pop(attr, None)
            self.SIDM.__dict__.pop(attr, None)
        try: 
            self.notes.writeConstants()
            self.notes.writeProperties()
        except AttributeError:
            pass

        
    @property
    def H0(self) -> u.quantity.Quantity:
        """
        Hubble constant at z = 0.
        Relevant for defining the surrounding cosmology.
        Usual unit: km / s / Mpc
        """
        if type(self.H0Input) is not u.quantity.Quantity:
            return self.H0Input * u.km/u.s/u.Mpc
        else:
            return self.H0Input
        
    @H0.setter
    def H0(self, __new):
        self.H0Input = __new
        #Need to delete cached values of all variables which depend on H0 when H0 changes
        for attr in ('h', 'rho_crit', 'rho200', 'R200', 'rs', 'V200', 'rho0', 'T200', 'transRadius', 'transDens', 'transMass', 'KingRadius', 'centralDens'):
            self.__dict__.pop(attr, None)
            self.NFW.__dict__.pop(attr, None)
            self.SIDM.__dict__.pop(attr, None)
        try: 
            self.notes.writeConstants()
            self.notes.writeProperties()
        except AttributeError:
            pass
        
    @property
    def scatteringParam(self) -> u.quantity.Quantity:
        """dependencies
        Measure for the rate of scattering events occurring within a given time and density. 
        Mathematically described by the quantity sigma / m.
        Usual unit: cm^2 / g
        """
        if type(self.scatteringParamInput) is not u.quantity.Quantity:
            return self.scatteringParamInput * u.cm**2/u.g
        else:
            return self.scatteringParamInput
        
    @scatteringParam.setter
    def scatteringParam(self, __new):
        self.scatteringParamInput = __new
        #Need to delete cached values of all variables which depend on scatteringParam when scatteringParam changes 
        for attr in ('transRadius', 'transDens', 'transMass', 'KingRadius', 'centralDens'):
            self.SIDM.__dict__.pop(attr, None)
        try: 
            self.notes.writeConstants()
            self.notes.writeProperties()
        except AttributeError:
            pass

    @property
    def haloAge(self) -> u.quantity.Quantity:
        """
        Age of the dark matter halo.
        Usually comparable to the age of the universe.
        Usual unit: Gyr
        """
        if type(self.haloAgeInput) is not u.quantity.Quantity:
            return self.haloAgeInput * u.Gyr 
        else:
            return self.haloAgeInput
        
    @haloAge.setter
    def haloAge(self, __new):
        self.haloAgeInput = __new
        #Need to delete cached values of all variables which depend on haloAge when haloAge changes
        for attr in ('transRadius', 'transDens', 'transMass', 'KingRadius', 'centralDens'):
            self.SIDM.__dict__.pop(attr, None)
        try: 
            self.notes.writeConstants()
            self.notes.writeProperties()
        except AttributeError:
            pass

    @property
    def concentration(self) -> float:
        """
        Cosmological concentration parameter
        Usual unit: unitless
        """
        if type(self.conInput) is bool:
            return self.conMassRelation(self.conInput)
        else:
            return self.conInput
        
    @concentration.setter
    def concentration(self, __new):
        self.conInput = __new
        #Need to delete cached values of all variables which depend on concentration when concentration changes
        for attr in ('rs', 'delta', 'rho0', 'transRadius', 'transDens', 'transMass', 'KingRadius', 'centralDens'):
            self.__dict__.pop(attr, None)
            self.NFW.__dict__.pop(attr, None)
            self.SIDM.__dict__.pop(attr, None)
        try: 
            self.notes.writeConstants()
            self.notes.writeProperties()
        except AttributeError:
            pass

    @cached_property
    def h(self) -> u.quantity.Quantity:
        """
        Small h.
        Related to the Hubble constant (H0) by h = H0 / (100 km / s / Mpc)
        Usual unit: unitless
        """
        return (self.H0/(100*u.km/u.s/u.Mpc)).decompose()


    @cached_property
    def rho_crit(self) -> u.quantity.Quantity:
        """
        Critical density of the universe.
        Mathematically given by: rho_crit = 3 * H0^2 / (8 * pi * G),
        with H0 the Hubble constant and G Newton's gravitational constant.
        Usual unit: M_sun / Mpc^3
        """
        return ((3*self.H0**2)/(8*np.pi*c.G)).to(u.M_sun/u.Mpc**3)
    

    @cached_property
    def rho200(self) -> u.quantity.Quantity:
        """
        200 times the critical density of the universe.
        Defined as the mean density of the dark matter halo within a radius R200.
        Usual unit: M_sun / Mpc^3
        """
        return 200*self.rho_crit

    @cached_property
    def R200(self) -> u.quantity.Quantity:
        """
        The virial radius.
        Defined such that the spherical halo with a virial mass M200,
        has a mean density equal to 200*rho_crit within a radius of R200.
        Mathematically given by R200 = (3 * M200 / (4 * pi * 200 * rho_crit))^(1/3)
        Usual unit: Mpc
        """
        return self.calculateR200(self.M200)

    @cached_property
    def rs(self) -> u.quantity.Quantity:
        """
        Normalization radius for the NFW-profile.
        Determines the radius where the NFW-profile transitions from 
        following a rho ~ r^(-1) relation to a rho ~ r^(-3) relation.
        Mathematically given by rs = R200 / concentration.
        Usual unit: Mpc
        """
        return self.R200/self.concentration
    

    @cached_property
    def V200(self) -> u.quantity.Quantity:
        """
        Circular velocity at radius R200.
        Given by V200 = (G * M200 / R200)^(1/2).
        Usual unit: km / s
        """
        return np.sqrt(c.G*self.M200/self.R200).to(u.km/u.s)
    
    def calculateR200(self, M200: np.ndarray[u.quantity.Quantity | float]) -> np.ndarray[u.quantity.Quantity]:
        """
        Calculate the virial radius, R200, for a given virial mass, M200.
        Calculates R200 by R200 = (3 * M200 / (4 * pi * 200 * rho_crit))^(1/3),
        assumes rho_crit to be equal to model.rho_crit.

        This function was made to easily calculate R200 for any M200, outside of
        `HaloModel`, this way a new instance of `HaloModel` did not have to be made
        when calculating R200 for each M200 in a grid.

        Parameters
        ----------
        `M200` [M_sun]: The virial mass, M200.

        Returns
        -------
        `R200` [Mpc]: The virial radius, R200.
        """
        if type(M200) is not u.quantity.Quantity:
            return (((3*M200 * u.M_sun)/(4*np.pi*self.rho200))**(1/3)).to(u.Mpc)
        else:
            return (((3*M200)/(4*np.pi*self.rho200))**(1/3)).to(u.Mpc)

    def conMassRelation(self, relaxed: bool = True, log_sigma: float = 0.) -> float:
        """
        Concentration - Virial Mass relation
        For a given M200 calculates the associated concentration via empirical equations.
        Can be for a relaxed halo and a non-relaxed halo.

        Based on equations (4) and (5) in Neto et al. 2007.

        Parameters 
        ----------
        `relaxed`: Boolean wether to consider a relaxed halo or not, True corresponds to a relaxed halo.

        `log_sigma`: For relaxed halos, how many multiples of 0.1 to add or subtract from calculated log-scale concentration.

        Returns
        -------
        `concentration`: The determined concentration using the appropriate equation.
        """
        if relaxed:
            conc = (5.26*(self.M200/(10**14/self.h*u.M_sun))**(-0.10)).decompose().value
            conc = 10**(np.log10(conc) + log_sigma*0.1)
            return conc
        else:
            return (4.67*(self.M200/(10**14/self.h*u.M_sun))**(-0.11)).decompose().value

    def constantsSetter(self, dictionary: dict):
        """
        Set all the relevant base constants of the model in one go.
        Used when loading a model from a save-file.

        This function is usually called in conjunction with the `notes.getConstants()` function, 
        which generates the relevant dictionary.

        `dictionary` is a dictionary containing the values of the base parameters with their keys:
        Parameter, key
        M200: log(M200)
        concentration: concentration
        H0: H0
        scatteringParam: sigma/m
        haloAge: halo Age
        rhoMean: rhoMean
        mu: mu 
        phi: phi
        chi_rho: chi_rho
        chi_T: chi_T
        chi_mu: chi_mu
        """
        self.M200Input            = 10**dictionary["log(M200)"]
        self.conInput             = dictionary["concentration"]
        self.H0Input              = dictionary["H0"]
        self.scatteringParamInput = dictionary["sigma/m"]
        self.haloAgeInput         = dictionary["halo Age"]
        self.NFW.rhoMeanInput     = dictionary["rhoMean"]
        self.NFW.muInput          = dictionary["mu"]
        self.NFW.phiInput         = dictionary["phi"]
        self.EoS.chi_rho          = dictionary["chi_rho"]
        self.EoS.chi_T            = dictionary["chi_T"]
        self.EoS.chi_mu           = dictionary["chi_mu"]

    def load(self, directory: str):
        """
        Loads a model from a save file, in which the base parameters are given.

        Parameters
        ----------
        `directory` is a string pointing to the save file with the parameters.
        """
        path = directory
        if "/" not in directory:
            path = os.getcwd() + "/Simulations/" + directory + "/"
            

        with open(path + "notes.txt", "r") as file:
            lines = file.readlines()
            constants_index  = np.where(["Constants" in line for line in lines])[0][0]
            constants_string = lines[constants_index+1:constants_index+12]
            constants = {line[:line.find("=")-1]: str_convert(line[line.find("=")+2:line.find("[")]) for line in constants_string}

        self.constantsSetter(constants)

    def gridPath(self, grid : str) -> str:
        """
        Gives the path which points to the location of (yet to be) generated grid.
        This function takes into account the different folder created when scatteringParam != 0

        Parameters
        ----------
        `grid`: A string which is the name of the desired grid, without the .csv suffix.

        Returns
        -------
        `gridPath`: The path to the desired grid.csv-file.

        Examples
        --------
        >>> gridPath('G-Grid')
        name_of_model/G-Grid.csv
        >>> # this model has scatteringParam = 10
        >>> gridPath('G-Grid')
        name_of_model/SIDM/(sigma = 10)/G-Grid.csv
        """
        output_path = self.fullPath + f"{grid}.csv"
        if self.scatteringParam.value != 0: 
            os.makedirs(self.fullPath + f"SIDM/(sigma = {self.scatteringParam.value})", exist_ok=True)
            output_path = self.fullPath + f"SIDM/(sigma = {self.scatteringParam.value})/{grid}.csv"
        return output_path

class NFW:
    def __init__(self, haloModel: HaloModel, mu=0.6, phi=1.0, rhoMean=4.9310966e+09*u.M_sun/u.Mpc**3):
        """
        NEEDS DOCUMENTATION
        """
        self.model          = haloModel
        self.muInput        = mu
        self.phiInput       = phi
        self.rhoMeanInput   = rhoMean

        self.T0             = 10**4*u.K
        self.gamma0         = 0.54
        self.powerLawRho0   = (c.m_p*10**(-6)/u.cm**3).to(u.kg/u.cm**3)

        self.useT200_approx = True

    @property
    def rhoMean(self) -> u.quantity.Quantity:
        """
        Crude estimate of the mean baryonic density in the universe.
        In this project taken to be approximately equal to 10^(-6.7) cm^-3 Hydrogen nuclei.
        Usual unit: M_sun / Mpc^3
        """
        if type(self.rhoMeanInput) is not u.quantity.Quantity:
            return self.rhoMeanInput * u.M_sun/u.Mpc**3
        else:
            return self.rhoMeanInput
    
    @rhoMean.setter
    def rhoMean(self, __new):
        self.rhoMeanInput = __new
        #Need to delete cached values of all variables which depend on rhoMean when rhoMean changes
        for attr in ('logRhoMean'):
            self.__dict__.pop(attr, None)
        try: 
            self.model.notes.writeConstants()
            self.notes.writeProperties()
        except AttributeError:
            pass
    
    @property
    def mu(self) -> float:
        """
        Average molecular weight.
        Usual unit: unitless
        """
        return self.muInput
    
    @mu.setter
    def mu(self, __new):
        self.muInput = __new
        #Need to delete cached values of all variables which depend on mu when mu changes
        try:
            del self.T200
        except AttributeError:
            pass
        try: 
            self.model.notes.writeConstants()
            self.notes.writeProperties()
        except AttributeError:
            pass

    @property
    def phi(self) -> float:
        """
        Equation-of-State dependent variable which relates the 
        average kinetic energy per mass of particles (u) to the pressure (P) and density (rho).
        Given by u = phi * P/rho
        Usual unit: unitless
        """
        return self.phiInput
    
    @phi.setter
    def phi(self, __new):
        self.phiInput = __new
        #Need to delete cached values of all variables which depend on phi when phi changes
        try:
            del self.T200
        except AttributeError:
            pass
        try: 
            self.model.notes.writeConstants()
            self.notes.writeProperties()
        except AttributeError:
            pass
    
    @cached_property
    def logRhoMean(self) -> float:
        """
        Base-10 logarithmic version of the mean baryon density of the universe.
        Mathematically given by log_10(rho_mean / (M_sun / Mpc^3))
        Usual unit: unitless
        """
        return np.log10(self.rhoMean/(u.M_sun/u.Mpc**3))

    @cached_property
    def delta(self) -> float:
        """
        Normalization factor for the NFW profile
        Calculated such that within a radius R200 a mass of M200 is reached. 
        Mathematically given by delta = (200 / 3) * (concentration ^3) / (log_10(1 + concentration) - concentration / (1 + concentration))
        Usual unit: unitless
        """
        return (200/3)*(self.model.concentration**3)/(np.log(1+self.model.concentration)-self.model.concentration/(1+self.model.concentration))

    
    @cached_property
    def rho0(self) -> u.quantity.Quantity:
        """
        Short hand for the product of the variables delta and the critical density. 
        Functions as normalization factor for the NFW profile.
        Usual unit: M_sun / Mpc^3
        """
        return self.delta*self.model.rho_crit
    
    @cached_property
    def T200(self) -> u.quantity.Quantity:
        """
        Temperature at the virial radius R200.
        Calculated by T200 = mu * m_p * V200^2 / (2 * phi * k_B).
        Usual unit: Kelvin 
        """
        if self.useT200_approx:
            return (10**4)*u.K*(self.circularVelocity(self.model.R200)/(17*u.km/u.s))**2
        else:
            return self.temperature(self.model.R200)

    def rho(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        NFW-profile 
        Gives the dark matter density at a radius.

         Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing radii
        \t       to calculate the DM-density at.

        Returns
        -------
        `rho` [M_sun / Mpc^3]: The determined DM-densities
        """

        radii = radii.to(self.model.rs.unit)
        return self.rho0/((radii/self.model.rs)*(1+radii/self.model.rs)**2)
    
    def smallLimit(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Limit of the NFW-profile for small radii (r << r_s),
        where it roughly follows a rho ~ r^(-1) relation.
        
        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing radii
        \t       to calculate the DM-density at.

        Returns
        -------
        `smallLimit` [M_sun / Mpc^3]: The determined DM-densities
        """
        
        radii = radii.to(self.model.rs.unit)
        return self.rho0*(radii/self.model.rs)**(-1)

    def largeLimit(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Limit of the NFW-profile for large radii (r >> r_s),
        where it roughly follows a rho ~ r^(-3) relation.
        
        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing radii
        \t       to calculate the DM-density at.

        Returns
        -------
        `largeLimit` [M_sun / Mpc^3]: The determined DM-densities
        """
        
        radii = radii.to(self.model.rs.unit)
        return self.rho0*(radii/self.model.rs)**(-3)
    
    def enclosedMass(self, radii: np.ndarray[u.quantity.Quantity], normalized: bool = False) -> np.ndarray[u.quantity.Quantity]:
        """
        Enclosed dark matter mass within a radius.
        If wanted to be given as a fraction of the virial mass, set `normalized=True`
        
        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing radii
        \t       to calculate the enclosed DM masses at.
        
        `normalized`: A boolean determining if the outputted enclosed masses should be normalized
        \t            to M200 or not.

        Returns
        -------
        `enclosedMass` [M_sun, unitless]: The determined enclosed DM masses.

        """
        
        normalizedRadius = (radii/self.model.R200).decompose()
        top              = np.log(1+self.model.concentration*normalizedRadius) - self.model.concentration*normalizedRadius/(1+self.model.concentration*normalizedRadius)
        bottom           = np.log(1+self.model.concentration) - self.model.concentration/(1+self.model.concentration)
        if normalized:
            return top/bottom
        else:
            return self.model.M200*top/bottom
        
    def circularVelocity(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Circular velocity at a radius.
        Given by the equation V = (G * M / R)^(1/2),
        with G Newton's gravitational constant and M the enclosed mass within radius R.
        
        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing radii
        \t       to calculate the circular velocities.

        Returns
        -------
        `circularVelocity` [km / s]: The calculated circular velocities.
        """
        
        return np.sqrt(c.G*self.enclosedMass(radii)/radii).to(u.km/u.s)
    
    def temperature(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Gives the average temperature of the dark matter particles at a radius.
        Given by the equation: mu * m_p * V^2/(2 * phi * k_B),
        with mu the mean molecular weight, m_p the proton mass,
        V the circular velocity at the desired radius, phi the encodes the applied equation of state,
        and k_B is the boltzmann constant.
        
        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing radii
        \t       to calculate the temperatures at.

        Returns
        -------
        `temperature` [K]: The calculated temperatures.

        """
        
        return ((self.mu*c.m_p*self.circularVelocity(radii)**2)/(2*self.phi*c.k_B)).to(u.K)

    def gasDensity(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Gives the density of gas-mass at a radius, 
        for a power-law equation of state between the temperature and density.
        See equation (A11) of Benitez-Llambay et al. 2017.
        
        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing radii
        \t       to calculate the gas densities at.

        Returns
        -------
        `gasDensity` [M_sun / Mpc^3]: The calculated gas masses.
        """
        
        normRadius = (radii/self.model.R200).decompose()
        MassInt    = (1/normRadius)*np.log(1+self.model.concentration*normRadius)/(np.log(1+self.model.concentration) - self.model.concentration/(1+self.model.concentration))
        factor     = (2*self.phi*self.gamma0/(1+self.gamma0))*(self.T200/self.T0)*(self.powerLawRho0/self.rhoMean)**(self.gamma0)
        return self.rhoMean*(factor*MassInt+1)**(1/self.gamma0)
    
    def enclosedGasMass(self, radius: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Gas mass within a radius,
        as determined by a power-law equation of state between the temperature and density.
        See equation (A11) of Benitez-Llambay et al. 2017.
        
        Parameters
        ----------
        `radius`: The desired radius at which the enclosed gas mass is to be calculated.
        \t        Has to be an astropy quantity.

        Returns
        -------
        `enclosedGasMass` [M_sun]: The enclosed gas mass at the inserted radius.
        """
        
        radii     = np.linspace(1e-12*radius.unit, radius, 1000)
        densities = self.gasDensity(radii)
        integral  = np.trapz(x= radii, y= 4*np.pi*(radii**2)*densities)
        return integral
    
    def gravitationalAcceleration(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Gravitational acceleration towards the halo centre due to the enclosed dark matter mass at a radius.
        Given by the equation G * M / R^2, with G Newton's gravitational constant and 
        M the enclosed dark matter mass within a radius R.
        
        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing how far away from 
        \t       the halo centre the gravitational acceleration is to be calculated.

        Returns
        -------
        `gravAcc` [cm / s^2]: The gravitational acceleration due to the DM-mass within the given radii.
        """
        
        return (c.G*self.enclosedMass(radii)/radii**2).to(u.cm/u.s**2)
    
    def G_integral(self, r_norms: np.ndarray[float]) -> np.ndarray[float]:
        """
        Gives the value for the integral appearing in the definition of G(r), for an NFW-profile.
        
        Parameters
        ----------
        `r_norms`: A float or an array of floats, for which the G-integrals need to be calculated.
        \t         `r_norms` are the normalized radii, with respect to R200.

        Returns
        -------
        `G_integral` [unitless]: The calculated G-integral value(s) for the inserted r_norm(s).
        """
        return -(1/r_norms)*np.log(1+self.model.concentration*r_norms)/(np.log(1+self.model.concentration) - self.model.concentration/(1+self.model.concentration))

class EoS:
    def __init__(self,  haloModel: HaloModel,  powerLawIndices: np.ndarray | list = [1, 1, -1], 
                                               logTpolyCoeff:   np.ndarray | None = None, 
                                               logMuPolyCoeff:  np.ndarray | None = None,
                                               logrho_Data:     np.ndarray | None = None):
        """
        NEEDS DOCUMENTATION
        """
        self.model = haloModel
        self.chi_rho, self.chi_T, self.chi_mu = powerLawIndices
        self.logTpolyCoeff  = logTpolyCoeff  #log(T)-log(rho)  polynomial descending order
        self.logMuPolyCoeff = logMuPolyCoeff #log(mu)-log(rho) polynomial descending order
        self.logrho_Data    = logrho_Data    #log(rho)-range where the polynomials for log(T) and log(mu) are valid
        
        # (Extrap)
        self.a = 0.545375
        self.b = 11.9935
        self.c = 1.8713
        self.d = 2.3709

        # (Isothermal)
        self.a = 0.543124
        self.b = 11.7569
        self.c = 3.48244
        self.d = 4.18048

        if self.logTpolyCoeff is None: #If no relationship given, use the one from paper
            self.logn_Data, self.logT_Data = np.loadtxt("Data/Benitez-Llambay et al. 2017/n-T data (isothermal).txt", delimiter=",").T

            self.logrho_Data    = self.logn_Data + np.log10(1/u.cm**3 * c.m_p/(u.M_sun/u.Mpc**3)).value
            self.logTpolyCoeff  = np.load(os.getcwd() + "/Data/Benitez-Llambay et al. 2017/logT - logrho Polynomial Coefficients (extended).npy")[::-1]

        if self.logMuPolyCoeff is None: #If no relationship given, assume constant value given in NFW.mu
            self.logMuPolyCoeff = np.log10(self.model.NFW.mu)

    def integral_solution(self, logrhos):
        term1  = logrhos
        term2  = -np.log(np.exp(self.d*logrhos)+np.exp(self.b*self.d))/self.d
        hypGeo = hyp2f1(1, self.c/self.d, 1+self.c/self.d, -np.exp(self.d*(logrhos-self.b)))
        term3  = -np.exp(self.c*(logrhos-self.b))*hypGeo/self.c
        return self.a*(term1 + term2 +term3)

    def logT(self, logrhos: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculates the appropriate log(T) value(s) in K for a given log(rho) or array of log(rho).
        The log(T) values are calculated via a polynomial in terms of powers of log(rho), 
        the coefficients of which are given in the `logTpolyCoeff` variable.

        The polynomial needs to be valid over the entire `logrho_Data`-range,
        will raise an error if any inserted logrhos lie outside this range.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which the log(T) values are to be calculated.
        \t         `logrhos` can either be a single float or an array of floats.
        \t         Note that logrhos can have no unit, so make sure it has the same units as the polynomial before converting it to log(rho).
        
        Returns
        -------
        `logT`: The calculated log(T) value(s) for the inserted logrho(s).
        """
        #If _logrho falls within log(rho)-data range: Use the log(T)-polynomial
        if np.all((self.logrho_Data[0] <= logrhos) & (logrhos <= self.logrho_Data[-1])):
            return self.integral_solution(logrhos) - self.integral_solution(self.logrho_Data[0])
            #return np.interp(logrhos, xp=self.logrho_Data, fp=self.logT_Data)
            #return np.poly1d(self.logTpolyCoeff)(logrhos)
        else:
            raise ValueError("logT: The inserted log(rho) lies outside the log(rho)-data range")

    def logMu(self, logrhos: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculates the appropriate log(mu) value(s) for a given log(rho) or array of log(rho).
        The log(mu) values are calculated via a polynomial in terms of powers of log(rho), 
        the coefficients of which are given in the `logMuPolyCoeff` variable.

        The polynomial needs to be valid over the entire `logrho_Data`-range,
        will raise an error if any inserted logrhos lie outside this range.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which the log(mu) values are to be calculated.
        \t         `logrhos` can either be a single float or an array of floats.
        \t         Note that logrhos can have no unit, so make sure it has the same units as the polynomial before converting it to log(rho).
        
        Returns
        -------
        `logMu`: The calculated log(mu) value(s) for the inserted logrho(s).
        """
        if np.all((self.logrho_Data[0] <= logrhos) & (logrhos <= self.logrho_Data[-1])):
            return np.poly1d(self.logMuPolyCoeff)(logrhos)
        else:
            raise ValueError("logMu: The inserted log(rho) lies outside the log(rho)-data range")

    def logT_diff(self, logrhos: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculates the appropriate dlog(T)/dlog(rho) value(s) for a given log(rho) or array of log(rho).
        The log(T) values are calculated via a polynomial in terms of powers of log(rho), 
        the coefficients of which are given in the `logTpolyCoeff` variable.

        The polynomial needs to be valid over the entire `logrho_Data`-range,
        will raise an error if any inserted logrhos lie outside this range.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which the dlog(T)/dlog(rho) values are to be calculated.
        \t         `logrhos` can either be a single float or an array of floats.
        \t         Note that logrhos can have no unit, so make sure it has the same units as the polynomial before converting it to log(rho).
        
        Returns
        -------
        `logT_diff`: The calculated dlog(T)/dlog(rho) value(s) for the inserted logrho(s).
        """
        #If _logrho falls within log(rho)-data range: Use the log(T)-polynomial
        if np.all((self.logrho_Data[0] <= logrhos) & (logrhos <= self.logrho_Data[-1])):
            return self.a * (1-np.exp(self.c*(logrhos-self.b)))/(1+np.exp(self.d*(logrhos-self.b)))
            #return np.poly1d(self.logTpolyCoeff).deriv()(logrhos)
        else:
            raise ValueError("logT_diff: The inserted log(rho) lies outside the log(rho)-data range")

    def logMu_diff(self, logrhos: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculates the appropriate dlog(mu)/dlog(rho) value(s) for a given log(rho) or array of log(rho).
        The log(mu) values are calculated via a polynomial in terms of powers of log(rho), 
        the coefficients of which are given in the `logMuPolyCoeff` variable.

        The polynomial needs to be valid over the entire `logrho_Data`-range,
        will raise an error if any inserted logrhos lie outside this range.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which the dlog(mu)/dlog(rho) values are to be calculated.
        \t         `logrhos` can either be a single float or an array of floats.
        \t         Note that logrhos can have no unit, so make sure it has the same units as the polynomial before converting it to log(rho).
        
        Returns
        -------
        `logMu_diff`: The calculated dlog(mu)/dlog(rho) value(s) for the inserted logrho(s).
        """
        if np.all((self.logrho_Data[0] <= logrhos) & (logrhos <= self.logrho_Data[-1])):
            return np.poly1d(self.logMuPolyCoeff).deriv()(logrhos)
        else:
            raise ValueError("logMu_diff: The inserted log(rho) lies outside the log(rho)-data range")

    def T_over_Mu(self, logrhos: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculates the appropriate T/mu value(s) in K for a given log(rho) or array of log(rho).
        The T/mu values are calculated via polynomials in terms of powers of log(rho), 
        the coefficients of which are given in the `logTpolyCoeff` and `logMuPolyCoeff` variables.

        The polynomial needs to be valid over the entire `logrho_Data`-range,
        will raise an error if any inserted logrhos lie outside this range.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which the T/mu values are to be calculated.
        \t         `logrhos` can either be a single float or an array of floats.
        \t         Note that logrhos can have no unit, so make sure it has the same units as the polynomial before converting it to log(rho).
        
        Returns
        -------
        `T_over_mu`: The determined T/mu value(s) for the inserted logrho(s).
        """
        return 10**(self.logT(logrhos)-self.logMu(logrhos))

class densProfiles:
    def __init__(self, haloModel: HaloModel):
        """
        NEEDS DOCUMENTATION
        """
        self.model         = haloModel
    
    def F_log_rho_integrand(self, logrhos: np.ndarray[float]) -> np.ndarray[float]:
        """
        A function which returns the value for the integrand of the integral within
        the expression for F(rho) that has the `chi_rho` term in front.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which the integrand values are to be calculated.
        \t         `logrhos` can either be a single float or an array of floats.
        
        Returns
        -------
        `rho_integrand`: The calculated integrands
        """
        return self.model.EoS.T_over_Mu(logrhos)
    
    def F_log_T_integrand(self, logrhos: np.ndarray[float]) -> np.ndarray[float]:
        """
        A function which returns the value for the integrand of the integral within
        the expression for F(rho) that has the `chi_T` term in front.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which the integrand values are to be calculated.
        \t         `logrhos` can either be a single float or an array of floats.
        
        Returns
        -------
        `T_integrand`: The calculated integrands
        """
        return self.model.EoS.T_over_Mu(logrhos) * self.model.EoS.logT_diff(logrhos)
    
    def F_log_mu_integrand(self, logrhos: np.ndarray[float]) -> np.ndarray[float]:
        """
        A function which returns the value for the integrand of the integral within
        the expression for F(rho) that has the `chi_mu` term in front.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which the integrand values are to be calculated.
        \t         `logrhos` can either be a single float or an array of floats.
        
        Returns
        -------
        `mu_integrand`: The calculated integrands
        """
        return self.model.EoS.T_over_Mu(logrhos) * self.model.EoS.logMu_diff(logrhos)
    
    def G(self, r_norms: np.ndarray[float]) -> np.ndarray[float]:
        """
        A function which determines the value of G(r) function for a single value of r_norm
        or an array of r_norms. 

        This function has 2 pre-builtin version, one for a NFW DM-profile, and the other 
        for a SIDM DM-profile based of a piecewise-function (NFW, isothermal sphere).

        Parameters
        ----------
        `r_norms`: The with respect to R200 normalized radii where the value of G is to be calculated for.
        \t         r_norms can either be a float or an array of floats.

        Returns
        -------
        `G`: The G-value(s) of the inserted r_norm(s)
        """
        prefactor = -2*self.model.NFW.phi*(self.model.NFW.T200/self.model.NFW.mu)
        
        if self.model.scatteringParam.value == 0:
            integrals = self.model.NFW.G_integral(r_norms)
        else:
            radii_inside  = r_norms[r_norms * self.model.R200 <  self.model.SIDM.transRadius]
            radii_outside = r_norms[r_norms * self.model.R200 >= self.model.SIDM.transRadius]

            Gs_inside     = self.model.SIDM.G_integral(radii_inside) + self.model.NFW.G_integral(self.model.SIDM.transRadius/self.model.R200)
            Gs_outside    = self.model.NFW.G_integral(radii_outside)

            integrals = np.hstack((Gs_inside, Gs_outside))
            if integrals.size == 1: integrals = integrals[0]
        
        return prefactor*integrals
    
    def F(self, logrho: float, returnError: bool = False, returnAll = False) -> float:
        """
        A function which determines the value of F(rho) function for a single value of log(rho). 

        Parameters
        ---------
        `logrho`: The log(rho) value to determine F(rho) for. Note that log(rho) has no units,
        \t        so make sure it had the same units as logRhoMean and the log(rho)-values used to
        \t        define the polynomials for log(T) and log(mu), before it was converted to log(rho) from rho.
        `returnError`: Wether to give the error of the calculated value of F(rho). Should only be used when
        \t        calculating arbitrary values, not when determining entire grids.
        
        Returns
        -------
        `total_integral`: The F-value for the inserted log(rho).
        `total_error`: (only if returnError = True) The total error on `total_integral`.
        """
       
        integrals = []
        errors    = []
        for chi, integrand_func in zip([self.model.EoS.chi_rho, self.model.EoS.chi_T, self.model.EoS.chi_mu], 
                                       [self.F_log_rho_integrand, self.F_log_T_integrand, self.F_log_mu_integrand]):
            integral = 0
            error    = 0
            if chi != 0:
                evaluate = quad(integrand_func, self.model.NFW.logRhoMean, logrho, limit=1000, epsabs=1e-12)
                integral = chi * evaluate[0]
                error    = chi * evaluate[1]
            integrals.append(integral)
            errors.append(error)
        
        total_integral = np.log(10) * np.sum(integrals) * u.K
        total_error    = np.log(10) * np.sqrt(np.sum(np.power(errors, 2))) * u.K
        
        if returnAll:
            return np.log(10) * np.array(integrals)
        
        if returnError:
            return total_integral, total_error
        else:
            return total_integral

    def Equalizer(self, logrho: float) -> tuple:
        """
        A function which finds the corresponding r_norm value for the inputted logrho value
        by determining the F(rho) value for the inputted logrho and then solving for which
        r_norm the G(r) function gives this same value. 

        This function is only used for finding bugs and making sure everything works correctly.

        Parameters
        ----------
        `logrho`: The log(rho) value for which r_norm is to be found.

        Returns
        -------
        `r_norm`: The associated normalized radius for the given log(rho).
        
        `Fr`:     The F-value for the given log(rho).
        
        `Gr`:     The G-value for the determined `r_norm`.
        """
        Fr = self.F(logrho)
        
        r_norm = fsolve(lambda r_norm: (self.G(r_norm) - Fr), 1e-2)[0]
    
        return r_norm, Fr, self.G(r_norm)

    def F_GridCreator(self, logrhos: np.ndarray[float] | None = None) -> np.ndarray[float]:
        """
        A function which generates a grid of F-values for each inputted logrhos value and saves it 
        into a file named 'F-Grid.csv' in the directory of the model.

        Parameters
        ----------
        `logrhos`: The log(rho) values for which to calculate each value of F for.
        \t         logrhos can either be an array or None. If it is None, this function will use
        \t         the values given in `EoS.logrho_Data` instead. 
        """
        if logrhos is None:
           logrhos = self.model.EoS.logrho_Data

        valid = (self.model.EoS.logrho_Data[0] <= logrhos) & (logrhos <= self.model.EoS.logrho_Data[-1])

        if sum(~valid) > 0: 
            raise ValueError("Inputted logrhos go beyond the EoS.logrho_Data range")

        Fs = np.zeros(len(logrhos))
        for i, logrho in enumerate(logrhos):
            print(f"F_Grid: {(i+1):{len(str(len(logrhos)))}d}/{len(logrhos)}", end="\r")
            Fs[i] = (self.F(logrho)/u.K).value
        
        print()
        np.savetxt(self.model.fullPath + "F-Grid.csv", np.row_stack((logrhos, Fs)).T, delimiter=",", header=f"H0={self.model.H0.value}, rhoMean = {self.model.NFW.rhoMean.value}\n log(rho) [M_sun Mpc^-3], F(log(rho)) [K]")

    def G_GridCreator(self, M200s: np.ndarray[float | u.quantity.Quantity] = 10**np.linspace(8, 9.7, 40), r_norms: np.ndarray[float] = np.logspace(-4, 1, 200)):
        """
        A function which generates a grid of G-values for each inputted r_norm and M200 value and saves it 
        into a file named 'G-Grid.csv' in the directory of the model, or in 'SIDM/(sigma = scatteringParam.value)' if scatteringParam != 0.

        Parameters
        ----------
        `M200s`: The virial masses for which to calculate G-values for each r_norm for.
        
        `r_norms`: The with respect to R200 normalized radii to calculate each G-value for for each virial mass.
        
        `encMassFunc`: Which enclosed DM-Mass function to use in the calculation of G(r).
        """
        table         = np.zeros((len(r_norms), len(M200s)+1))
        table[:,0]    = r_norms

        M200s_powers  = []

        temp_model = HaloModel(directory=False)
        temp_model.constantsSetter(self.model.notes.getConstants())
        for i, M200 in enumerate(M200s):
            print(f"G_Grid: {(i+1):{len(str(len(M200s)))}d}/{len(M200s)}", end="\r")
            temp_model.M200 = M200

            M200s_powers.append(np.log10(temp_model.M200/u.M_sun).value)
            table[:,i+1] = (temp_model.densProfile.G(r_norms)/u.K).value

        print()
        self.model.notes.writeM200s_powers(M200s_powers)

        np.savetxt(self.model.gridPath('G-Grid'),table, delimiter=",", header=f"G-value for r_norm = r/R200 vs log(M200) [M_sun], phi={self.model.NFW.phi}, mu={self.model.NFW.mu}, c={self.model.conInput}\n r_norm, {', '.join([str(power) for power in M200s_powers])}")

    def dens_GridCreator(self, F_path: str | None = None, G_path: str | None = None):
        """
        A function which generates a grid of gas density-values for each r_norm and M200 value used in the G-Grid and saves it 
        into a file named 'Density Profiles.csv' in the directory of the model, or in 'SIDM/(sigma = scatteringParam.value)' if scatteringParam != 0.

        The gas density values are calculated by taking the F- and G-Grids and linearly interpolating the G-values between the 
        (log(rho), F) datapoints to get a log(rho) value for each r_norm in the G-Grid.

        Parameters
        ----------
        `F_path`: A string which contains the path to the F-Grid.csv file to be used in generating the gas density profiles.
        \t        Alternatively, if set to None, the function will assume F-Grid.csv to be located within the directory of the model.
        
        `G_path`: A string which contains the path to the G-Grid.csv file to be used in generating the gas density profiles.
        \t        Alternatively, if set to None, the function will assume G-Grid.csv to be located within the directory of the model,
        \t        or in 'SIDM/(sigma = scatteringParam.value)' if scatteringParam != 0.
        """
        if F_path is None: F_path = self.model.fullPath + "F-Grid.csv"
        
        if G_path is None: 
            if self.model.scatteringParam.value == 0:
                G_path = self.model.fullPath + "G-Grid.csv"
            else:
                G_path = self.model.fullPath + f"SIDM/(sigma = {self.model.scatteringParam.value})/G-Grid.csv"

        if F_path != self.model.fullPath + "F-Grid.csv":
            os.system(f'cp "{F_path}" "{self.model.fullPath + "F-Grid.csv"}"')
            self.model.notes.write(f"Copied F-Grid from {F_path}\n")

        if G_path not in [self.model.fullPath + "G-Grid.csv", self.model.fullPath + f"SIDM/(sigma = {self.model.scatteringParam.value})/G-Grid.csv"]:
            copy_to = self.model.gridPath('G-Grid')
            
            os.system(f'cp "{G_path}" "{copy_to}"')
            self.model.notes.write(f"Copied G-Grid from {G_path}\n")

        M200s_powers = self.model.notes.getM200s_powers()
        constants    = self.model.notes.getConstants()

        F_Grid = np.loadtxt(F_path, delimiter=",")
        G_Grid = np.loadtxt(G_path, delimiter=",")

        r_norms = G_Grid[:,0]
        logrhos = F_Grid[:,0]
        Gs      = G_Grid[:,1:]
        Fs      = F_Grid[:,1]

        densprofiles      = np.zeros((len(r_norms), len(M200s_powers)+1))
        densprofiles[:,0] = r_norms

        for i in range(len(M200s_powers)):
            print(f"dens_Grid: {(i+1):{len(str(len(M200s_powers)))}d}/{len(M200s_powers)}", end="\r")
            densprofiles[:,i+1] = 10**np.interp(Gs[:,i], Fs[Fs > 0], logrhos[Fs > 0])
        
        print()
        np.savetxt(self.model.gridPath('Density Profiles'), densprofiles, delimiter=",", header=f"rho [M_sun/Mpc3] for r_norm = r/R200 vs log(M200) [M_sun]\n phi={constants['phi']}, mu={constants['mu']}, c={constants['concentration']}, H0={constants['H0']}, rhoMean={constants['rhoMean']} \n r_norm, {', '.join([str(power) for power in M200s_powers])}")

    def T_GridCreator(self):
        """
        A function which generates a grid of temperature-values for each r_norm and M200 value used in the G-Grid and saves it 
        into a file named 'T Profiles.csv' in the directory of the model, or in 'SIDM/(sigma = scatteringParam.value)' if scatteringParam != 0.

        The temperature values are calculated by taking the previously generated gas density values, located in "Density Profiles.csv", 
        and inserting them in the previously used log(rho)-log(T) polynomial.
        """
        M200s_powers = self.model.notes.getM200s_powers()
        
        dens_path = self.model.gridPath('Density Profiles')
        
        densGrid = np.loadtxt(dens_path, delimiter=",")
    
        radii = densGrid[:,0]
        denss = densGrid[:,1:]
        table = np.zeros((len(radii), len(M200s_powers)+1))
        
        table[:,0] = radii

        for i in range(len(M200s_powers)):
            print(f"T_Grid: {(i+1):{len(str(len(M200s_powers)))}d}/{len(M200s_powers)}", end="\r")
            table[:,i+1] = 10**self.model.EoS.logT(np.log10(denss[:,i]))

        print()
        np.savetxt(self.model.gridPath('T Profiles'), table, delimiter=",", header=f"T [K] for r_norm = r/R200 vs log(M200) [M_sun]\n r_norm, {', '.join([str(power) for power in M200s_powers])}")


    def gasMassCalculator(self, M200s_powers: np.ndarray[float] | None = None, r_mask: tuple =(0, 1)) -> np.ndarray[u.quantity.Quantity]:
        """
        Calculates the gas Mass of a halo for each M200 given in `M200s_powers`, within a radius-range of `r_mask[0]` to `r_mask[1]`.

        This done by taking the gas density profiles located in "Density Profiles.csv" and spherically integrating them from `r_mask[0]` to `r_mask[1]`,
        for each M200 in M200s_powers.

        Parameters
        ----------
        `M200s_powers`: The base-10 exponent giving the desired M200s to calculate the gas mass for.
        \t              Can either be a single float or an array of floats.
        
        `r_mask`: The with respect to R200 normalized radius range between which to integrate.
        \t        `r_mask` must be a tuple (or list) of size 2, giving the beginning and endpoint of the integration range.

        Returns
        -------
        `M_gass` [M_sun]: An array containing all the determined gas masses.
        """
        allM200s_powers = self.model.notes.getM200s_powers()
        
        if M200s_powers is None:
            M200s_powers = allM200s_powers
        
        if type(M200s_powers) is list:
            M200s_powers = np.array(M200s_powers)

        if type(M200s_powers) is float:
            M200s_powers = np.array([M200s_powers])
        
        gasProfiles = np.loadtxt(self.model.gridPath('Density Profiles'), delimiter=",")
        r_norms     = gasProfiles[:,0]
        profiles    = gasProfiles[:,1:]

        mask  = (r_mask[0] <= r_norms ) & (r_norms <= r_mask[1])
        R200s = self.model.calculateR200(10**M200s_powers*u.M_sun)

        r_norms  = r_norms[mask]
        profiles = profiles[mask][:,np.searchsorted(allM200s_powers, M200s_powers)]

        physical_r   = r_norms.reshape((1,len(r_norms))) * R200s.reshape((len(R200s),1))
        physical_rho = (profiles * u.M_sun/u.Mpc**3).T
        integrands   = 4 * np.pi * physical_rho * (physical_r ** 2)
        return np.trapz(x = physical_r, y=integrands)


    def generateAllGrids(self, Fparams:    np.ndarray[float] | None | bool = None, 
                               Gparams:    list[np.ndarray]                = [10**np.linspace(8, 9.7, 40), np.logspace(-4, 1, 200)], 
                               densParams: list[str | None]                = [None, None]):
        """
        A function for convenience.
        Determines the F-Grid, G-Grid, gas density Grid, and temperature-Grid in one go, in that order.

        Parameters
        ----------
        `Fparams`: The log(rho) range to integrate over, if set to None will use the data given in `EoS.logrho_Data`.
        \t         Alternatively, `Fparams` can equal `False`, doing so will skip the calculation of F-Grid. Useful
        \t         if F-Grid remains constant.

        `Gparams`: A list of length 3 containing the M200 values (array of float),
        \t         the normalized radii (array of floats), and which enclosed mass function to use (string or function).

        `densParams`: A list of length 2 containing the paths to the F-Grid.csv and G-Grid.csv files to be used for
        \t            determining the density profiles. If left as [None, None] will use the files given in the directory of the model.
        \t            Or in 'SIDM/(sigma = scatteringParam.value)' if scatteringParam != 0, for the G-Grid.csv.
        """
        if type(Fparams) is np.ndarray or Fparams is None: self.F_GridCreator(Fparams)
        self.G_GridCreator(*Gparams)
        self.dens_GridCreator(*densParams)
        self.T_GridCreator()

class Hydrogen_Profile:
    def __init__(self, haloModel: HaloModel):
        """
        NEEDS DOCUMENTATION
        """
        #HI-profile, based on Rahmati et al. (2013)
        self.model = haloModel

        #Assumed values from paper
        #Table 2 (z=0) (HM01)
        self.Gamma_UVB = 8.34*10**(-14)/u.s 

        #                Table A2            #Table A1 (z=0)
        self.n0        = 10**(-2.56)/u.cm**3 #10**(-2.94)/u.cm**3
        self.alpha1    = -1.86               #-3.98
        self.alpha2    = -0.51               #-1.09
        self.beta      = 2.83                #1.29
        self.oneMinusF = 0.99                #0.99
    
    def photoIonizationRate(self, n_Hs: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Calculates the photo ionization rate for given hydrogen nuclei number density/densities.
        This is an intermediate step in calculating the number density of neutral hydrogen.

        Parameters
        ----------
        `n_Hs`: An astropy quantity or an array of astropy quantities representing the hydrogen nuclei number density.

        Returns
        -------
        `photoIonizationRate` [1 / s]: The determined photo ionization rate for each inserted hydrogen nuclei number densities.
        """
        ratio = self.oneMinusF*(1+(n_Hs/self.n0)**self.beta)**self.alpha1 + (1-self.oneMinusF)*(1+n_Hs/self.n0)**self.alpha2
        return self.Gamma_UVB*ratio
    
    def alpha_A(self, Ts: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Determines the value(s) of the alpha_A function originally given in Hui & Gnedin (1997) for the inserted
        temperature(s). This is an intermediate step in calculating the number density of neutral hydrogen.

        Parameters
        ----------
        `Ts`: An astropy quantity or an array of astropy quantities representing temperatures.

        Returns
        -------
        `alpha_A` [cm^3 / s]: The calculated values of the alpha_A function for the inserted temperatures.
        """
        lambda_param = (315614 * u.K)/Ts
        return (1.269*10**(-13)*u.cm**3/u.s)*(lambda_param**(1.503)/(1+(lambda_param/0.522)**0.47)**1.923)

    def Lambda_T(self, Ts: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Determines the value(s) of the Lambda_T function originally given in Theuns et al. (1998) for the inserted
        temperature(s). This is an intermediate step in calculating the number density of neutral hydrogen.

        Parameters
        ----------
        `Ts`: An astropy quantity or an array of astropy quantities representing temperatures.

        Returns
        -------
        `Lambda_T` [cm^3 / s]: The calculated values of the Lambda_T function for the inserted temperatures.
        """
        return (1.17*10**(-10)*u.cm**3/u.s)*(np.sqrt(Ts/u.K)*np.exp(-157809*u.K/Ts))/(1+np.sqrt(Ts/(10**5*u.K)))

    def HIDensity(self, n_Hs: np.ndarray[u.quantity.Quantity], Ts: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Determines the neutral hydrogen number densities for the inserted hydrogen nuclei number densities and temperatures,
        as by the process described in Appendix A2 of Rahmati et al. (2013)

        Parameters
        ----------
        `n_Hs`: An astropy quantity or an array of astropy quantities representing the hydrogen nuclei number density.

        `Ts`: An astropy quantity or an array of astropy quantities representing temperatures.

        Returns
        -------
        `HIDensity` [1 / cm^3]: The calculated neutral hydrogen number densities.
        """
        _alpha_A, _Lambda_T = self.alpha_A(Ts), self.Lambda_T(Ts)
        A = _alpha_A + _Lambda_T
        B = 2*_alpha_A + self.photoIonizationRate(n_Hs)/n_Hs + _Lambda_T
        C = _alpha_A
        return n_Hs * (B-np.sqrt(B**2-4*A*C))/(2*A)
    
    def nHI_GridCreator(self):
        """
        A function which generates a grid of neutral hydrogen number density values for each r_norm and M200 value used in the G-Grid and saves it 
        into a file named 'HI-Density Profiles.csv' in the directory of the model, or in 'SIDM/(sigma = scatteringParam.value)' if scatteringParam != 0.

        The neutral hydrogen number density values are calculated using the gas density grid in 'Gas Profiles.csv' 
        and the temperature grid in 'T Profiles.csv' and inserting the corresponding value for each r_norm and M200 into
        the functions following the procedure described in Rahmati et al. (2013).
        """
        M200s_powers = self.model.notes.getM200s_powers()

        gasProfiles  = np.loadtxt(self.model.gridPath('Density Profiles'), delimiter=",")
        tempProfiles = np.loadtxt(self.model.gridPath('T Profiles'), delimiter=",")
        
        radii        = gasProfiles[:,0]
        profiles     = gasProfiles[:,1:]
        temperatures = tempProfiles[:,1:]

        table = np.zeros((len(radii), len(M200s_powers)+1))
        table[:,0] = radii

        temp_model = HaloModel(directory=False)
        temp_model.constantsSetter(self.model.notes.getConstants())
        for i, M200_pow in enumerate(M200s_powers):
            temp_model.M200 = 10**M200_pow
            print(f"nHI_Grid: {(i+1):{len(str(len(M200s_powers)))}d}/{len(M200s_powers)}", end="\r")
            rhos          = profiles[:,i] * u.M_sun/u.Mpc**3
            nHs           = (rhos/c.m_p).to(1/u.cm**3)

            Ts            = temperatures[:,i]*u.K
            table[:,i+1]  = temp_model.HProfile.HIDensity(nHs, Ts)

        print()
        np.savetxt(self.model.gridPath('HI-Density Profiles'), table, delimiter=",", header=f"n_HI [1/cm3] for r_norm = r/R200 vs log(M200) [M_sun]\n r_norm, {', '.join([str(power) for power in M200s_powers])}")


    def NHI_GridCreator(self):
        """
        A function which generates a grid of neutral hydrogen column density values for each r_norm and M200 value used in the G-Grid and saves it 
        into a file named 'HI-Column Density Profiles.csv' in the directory of the model, or in 'SIDM/(sigma = scatteringParam.value)' if scatteringParam != 0.

        The neutral hydrogen column density values are calculated using the number density grid in 'HI-Density Profiles.csv'
        and evaluating an integral for each M200 and normalized radius.
        """
        M200s_powers = self.model.notes.getM200s_powers()

        HIdens_path = self.model.gridPath('HI-Density Profiles')

        HIProfiles   = np.loadtxt(HIdens_path, delimiter=",")
        radii        = HIProfiles[:,0].reshape((len(HIProfiles[:,0]),1))
        profiles     = HIProfiles[:,1:]

        zs = np.linspace(-1, 1, 1000).reshape((1,1000))

        reduced_radii = np.sqrt(radii**2 + zs**2)

        table = np.zeros((len(radii), len(M200s_powers)+1))
        table[:,0] = HIProfiles[:,0]

        for i, M200_pow in enumerate(M200s_powers):
            print(f"NHI_Grid: {(i+1)}/{len(M200s_powers)}", end="\r")
            temp_R200 = self.model.calculateR200(10**M200_pow * self.model.M200.unit)
            
            HIProfile = profiles[:,i]/u.cm**3
            
            nHI_values = np.interp(x = reduced_radii, xp = radii.reshape(HIProfile.shape), fp=HIProfile)
            
            table[:, i+1] = (np.trapz(x = zs, y = nHI_values)* temp_R200).to(1/u.cm**2)
            

        print()
        np.savetxt(self.model.gridPath('HI-Column Density Profiles'), table, delimiter=",", header=f"N_HI [1/cm2] for r_norm = r/R200 vs log(M200) [M_sun]\n r_norm, {', '.join([str(power) for power in M200s_powers])}")

    def generateAllGrids(self):
        """
        A function for convenience.
        Determines the neutral hydrogen number density grid and column density grid in one go, in that order.
        """
        self.nHI_GridCreator()
        self.NHI_GridCreator()

    def HIMassCalculator(self, M200s_powers: np.ndarray[float] | None = None, r_mask: tuple =(0, 1)) -> np.ndarray[u.quantity.Quantity]:
        """
        Calculates the HI Mass of a halo for each M200 given in `M200s_powers`, within a radius-range of `r_mask[0]` to `r_mask[1]`.

        This done by taking the HI density profiles located in "HI-Density Profiles.csv" and spherically integrating them from `r_mask[0]` to `r_mask[1]`,
        for each M200 in M200s_powers.

        Parameters
        ----------
        `M200s_powers`: The base-10 exponent giving the desired M200s to calculate the HI mass for.
        \t              Can either be a single float or an array of floats.
        
        `r_mask`: The with respect to R200 normalized radius range between which to integrate.
        \t        `r_mask` must be a tuple (or list) of size 2, giving the beginning and endpoint of the integration range.

        Returns
        -------
        `M_gass` [M_sun]: An array containing all the determined HI masses.
        """
        allM200s_powers = self.model.notes.getM200s_powers()
        
        if M200s_powers is None:
            M200s_powers = allM200s_powers
        
        if type(M200s_powers) is list:
            M200s_powers = np.array(M200s_powers)

        if type(M200s_powers) is float:
            M200s_powers = np.array([M200s_powers])
        
        HIProfiles = np.loadtxt(self.model.gridPath('HI-Density Profiles'), delimiter=",")
        r_norms     = HIProfiles[:,0]
        profiles    = HIProfiles[:,1:]

        mask  = (r_mask[0] <= r_norms ) & (r_norms <= r_mask[1])
        R200s = self.model.calculateR200(10**M200s_powers*u.M_sun)

        r_norms  = r_norms[mask]
        profiles = profiles[mask][:,np.searchsorted(allM200s_powers, M200s_powers)]

        physical_r   = r_norms.reshape((1,len(r_norms))) * R200s.reshape((len(R200s),1))
        physical_rho = (profiles * c.m_p/u.cm**3).T
        integrands   = 4 * np.pi * physical_rho * (physical_r ** 2)
        return np.trapz(x = physical_r, y=integrands).to(u.M_sun)

class selfInteractingDarkMatter:
    def __init__(self, haloModel: HaloModel):
        """
        NEEDS DOCUMENTATION
        Robertson et al. (2021)
        Łokas & Mamon (2001)
        Galaxy Formation and Evolution (Houjun Mo, Frank van den Bosch, Simon White)
        """
        self.model = haloModel
    
    @cached_property
    def transRadius(self) -> u.quantity.Quantity:
        """
        The radius at which the DM-profile transitions from SIDM to NFW.

        The transition radius is defined as the radius where the number of DM interactions
        per age of the halo is approximately equal to unity, as by Robertson et al. (2021).

        Will equal 0 if scatteringParam = 0.

        Usual unit: Mpc
        """
        if self.model.scatteringParam.value != 0:
            return fsolve(lambda a: self.interactionRate(a*self.model.rs)*self.model.haloAge -1, 10**np.min([np.log10(self.model.scatteringParam.value)-1,0]))[0] * self.model.rs
        else:
            return 0*self.model.R200.unit

    @cached_property
    def transDens(self) -> u.quantity.Quantity:
        """
        The density at the transition radius.

        Calculated by inserting the transition radius into the NFW-profile.

        Necessary to calculate so the SIDM profile transitions smoothly.

        Will equal 0 if scatteringParam = 0.
        
        Usual unit: M_sun / Mpc^3
        """
        if self.model.scatteringParam.value != 0:
            return self.model.NFW.rho(self.transRadius)
        else:
            return 0*self.model.rho_crit.unit
    
    @cached_property
    def transMass(self) -> u.quantity.Quantity:
        """
        The enclosed DM mass at the transition radius.
        
        Calculated by integrating the NFW profile up to the transition radius.

        Necessary to calculate so the SIDM profile also encloses this mass at the transition radius.

        Will equal 0 if scatteringParam = 0.

        Usual unit: M_sun
        """
        if self.model.scatteringParam.value != 0:
            return self.model.NFW.enclosedMass(self.transRadius)
        else:
            return 0*self.model.M200.unit
    
    @cached_property
    def KingRadius(self) -> u.quantity.Quantity:
        """
        The radius which characterizes where the isothermal model transitions from
        the small radius approximation to the large radius approximation,
        as by Galaxy Formation and Evolution (Houjun Mo, Frank van den Bosch, Simon White).

        Necessary to calculate for the small radius approximation and central density.

        Will equal 0 if scatteringParam = 0.

        Usual unit: Mpc
        """
        if self.model.scatteringParam.value != 0:
            return np.abs(fsolve(self.r0Finder, 1)[0]*self.transRadius)
        else:
            return 0*self.model.R200.unit
    
    @cached_property
    def centralDens(self) -> u.quantity.Quantity:
        """
        The central density of the isothermal model,
        as by Galaxy Formation and Evolution (Houjun Mo, Frank van den Bosch, Simon White). 

        Necessary to calculate for the small radius approximation.

        Will equal 0 if scatteringParam = 0.

        Usual unit: M_sun / Mpc^3
        """
        if self.model.scatteringParam.value != 0:
            return self.centralDensFinder(self.KingRadius)
        else:
            return 0*self.model.rho_crit.unit
    
    def oneDimensionalVelocityDispersion(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        The one dimensional velocity dispersion (sigma_1D) at a given radius, 
        as originally given in Łokas & Mamon (2001).

        Necessary to calculate to determine the transition radius.

        Parameters 
        ----------
        `radii`: An astropy quantity or an array of astropy quantities,
        \t       for which the one dimensional velocity dispersion are to be calculated.

        Returns
        -------
        `sigma_1D` [km / s]: The one dimensional velocity dispersion(s) for the given radius/radii.
        """
        x   = (radii/self.model.rs).decompose().value
        g   = 1/(np.log(1+self.model.concentration) - self.model.concentration/(1+self.model.concentration))
        Li2 = spence(1+x)
        prefactor = 0.5*g*self.model.concentration*x*((1+x)**2)*(c.G*self.model.M200/self.model.R200)
        sum_part = np.pi**2 - np.log(x) - 1/x - 1/(1+x)**2 - 6/(1+x) + (1+1/x**2-4/x-2/(1+x))*np.log(1+x)+3*(np.log(1+x)**2)+6*Li2
        return np.sqrt(prefactor * sum_part).to(u.km/u.s)
    
    def meanVpair(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        The mean pairwise-velocity (<v_pair(r)>) at a radius, 
        given by <v_pair> = (4/sqrt(pi))*sigma_1D, as by Robertson et al. (2021).

        Necessary to calculate to determine the transition radius.

        Parameters 
        ----------
        `radii`: An astropy quantity or an array of astropy quantities,
        \t       for which the one mean pairwise-velocities are to be calculated.

        Returns
        -------
        `v_pair` [km / s]: The mean pairwise-velocities for the given radii.
        """
        return (4/np.sqrt(np.pi))*self.oneDimensionalVelocityDispersion(radii)
    
    def interactionRate(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        The SIDM interaction rate at given radius/radii, 
        give by Gamma = scatteringParam * <v_pair(r)> * rho(r), as by Robertson et al. (2021).

        Necessary to calculate to determine the transition radius.

        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities,
        \t       for which the interaction rates are to be calculated.

        Returns
        -------
        `Gamma` [1 / Gyr]: The interaction rates for the given radii.
        """
        return (self.model.scatteringParam * self.meanVpair(radii) * self.model.NFW.rho(radii)).to(1/self.model.haloAge.unit)
    
    def centralDensFinder(self, r0s: np.ndarray[u.quantity.Quantity]) -> u.quantity.Quantity:
        """
        Function to find the central density of the SIDM-profile for a given King radius.

        Could be used to calculate the "central density" for any arbitrary King radius,
        but its main function is to help determine the King radius of the SIDM-profile,
        and calculate the central density from that.

        Parameters
        ----------
        `r0s`: The King radius/radii to calculate the central density/densities for.
        \t     Can be an astropy quantity or an array of astropy quantities.

        Returns
        -------.
        `central density` [M_sun / Mpc^3]: The calculated central densities for the inserted radii.
        """
        return self.transDens*(1+(self.transRadius/r0s)**2)**(3/2)
    
    
    def r0Finder(self, r_norm: float) -> u.quantity.Quantity:
        """
        Function to find the King radius of the SIDM-profile.

        Could be used to calculate the "King radius" for any arbitrary normalized radius,
        but its main function is to calculate the King radius for the SIDM model.

        Parameters
        ----------
        `r_norm`: A normalized radius with respect to the transition radius.

        Returns
        -------
        `encMass - transMass`: The transition mass subtracted from the determined enclosed mass,
        \t                     if it equals 0, the inputted isothermal profile and NFW profile
        \t                     have the same enclosed mass within the transition radius,
        \t                     showing that the inputted `r_norm` * transition Radius is the King Radius.
        """
        r0 = r_norm * self.transRadius
        encMass =  4*np.pi*self.centralDensFinder(r0)*r0**3*(np.arcsinh(1/r_norm)-(1/r_norm)/np.sqrt(1+(1/r_norm)**2))
        return encMass - self.transMass
    
    def isoThermSmallr(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        A function which calculates the DM density as by the small radius approximation for an isothermal sphere,
        as given by Galaxy Formation and Evolution (Houjun Mo, Frank van den Bosch, Simon White).

        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities resembling the radii at which
        \t       to calculate the DM densities.

        Returns 
        -------
        `DM densities` [M_sun / Mpc^3]: The calculated dark matter densities for the inserted radii.
        """
        return self.centralDens * (1 + (radii/self.KingRadius)**2)**(-3/2)

    def isoEnclosedMassSmallr(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        A function which calculates the enclosed DM-mass as by the small radius approximation for an isothermal sphere,
        calculated by integrating the small radius approximation.

        Will return an array of zeros (length equal to radii) if scatteringParam = 0.

        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities resembling the radii at which
        \t       to calculate the enclosed DM masses.

        Returns 
        -------
        `enclosed masses` [M_sun]: The calculated enclosed dark matter masses for the inserted radii.
        \t                         Or an array of zeros if scatteringParam = 0.
        """
        if self.model.scatteringParam.value != 0:
            r_norm = (radii/self.KingRadius).decompose()
            return 4*np.pi*self.centralDens * (self.KingRadius**3) * (np.arcsinh(r_norm).value - r_norm/np.sqrt(1+r_norm**2))
        else:
            return np.zeros(np.array(radii).shape)*self.model.M200.unit
        
    def SIDMprofile(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        The composite SIDM density-profile, which equals the isothermal model within the transition radius,
        and the NFW-profile outside the transition radius.

        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities resembling the radii at which
        \t       to calculate the DM densities.

        Returns
        -------
        `DM densities` [M_sun / Mpc^3]: The calculated DM densities for the inserted radii.
        """
        radii_inside  = radii[radii <  self.transRadius]
        radii_outside = radii[radii >= self.transRadius]

        dens_inside   = self.isoThermSmallr(radii_inside)
        dens_outside  = self.model.NFW.rho(radii_outside)

        denss = np.hstack((dens_inside, dens_outside))
        if denss.size == 1: denss = denss[0]

        return denss

    def profileEnclosedMass(self, radii: np.ndarray[u.quantity.Quantity], normalized: bool = False) -> np.ndarray[u.quantity.Quantity | float]:
        """
        The composite SIDM enclosed mass-profile, which equals the isothermal model within the transition radius,
        and the NFW-profile outside the transition radius.

        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities resembling the radii at which
        \t       to calculate the enclosed DM masses.

        Returns
        -------
        `DM masses` [M_sun]: The calculated enclosed DM masses for the inserted radii.
        """
        radii_inside  = radii[radii <  self.transRadius]
        radii_outside = radii[radii >= self.transRadius]

        mass_inside   = self.isoEnclosedMassSmallr(radii_inside)
        mass_outside  = self.model.NFW.enclosedMass(radii_outside)

        mass = np.hstack((mass_inside, mass_outside))
        if mass.size == 1: mass = mass[0]

        if not normalized:
            return mass
        else:
            return mass/self.model.M200
        
    def gravitationalAcceleration(self, radii: np.ndarray[u.quantity.Quantity]) -> np.ndarray[u.quantity.Quantity]:
        """
        Gravitational acceleration towards the halo centre due to the enclosed dark matter mass at a radius.
        Given by the equation G * M / R^2, with G Newton's gravitational constant and 
        M the enclosed dark matter mass within a radius R.
        
        Parameters
        ----------
        `radii`: An astropy quantity or an array of astropy quantities representing how far away from 
        \t       the halo centre the gravitational acceleration is to be calculated.

        Returns
        -------
        `gravAcc` [cm / s^2]: The gravitational acceleration due to the DM-mass within the given radii.
        """
        return (c.G*self.profileEnclosedMass(radii)/radii**2).to(u.cm/u.s**2)
    
    def G_integral(self, r_norms: np.ndarray[float]) -> np.ndarray[float]:
        """
        Gives the value for the integral appearing in the definition of G(r), for an SIDM-profile.
        Note that the value of G for a SIDM-profile is also dependent on the variant of G for 
        a NFW profile, since the integral is calculated from infinity to some r_norm and
        the isothermal profile is only valid inside the transition radius.
        Hence, this function gives the value of G as if the integral was calculated from 
        the transition radius to the desired r_norms.
        
        Parameters
        ----------
        `r_norms`: A float or an array of floats, for which the G-integrals need to be calculated.
        \t         `r_norms` are the normalized radii, with respect to R200.

        Returns
        -------
        `G_integral` [unitless]: The calculated G-integral value(s) for the inserted r_norm(s).
        """
        constant  = -(4*np.pi*self.centralDens*(self.KingRadius**3)/self.model.M200).decompose()
        scale_fac = (self.model.R200/self.KingRadius).decompose()
        outside   = np.arcsinh(self.transRadius/self.KingRadius)/(self.transRadius/self.model.R200).decompose()
        return  constant*(np.arcsinh(scale_fac*r_norms)/r_norms - outside).value

class Notes:
    def __init__(self, haloModel: HaloModel):
        """
        NEEDS DOCUMENTATION
        """
        self.model = haloModel
        if not os.path.exists(self.model.fullPath+"notes.txt"): self.writeTimeOfCreation()
        self.writeConstants()
        self.writeProperties()

    def writeTimeOfCreation(self):
        """
        Function which writes the time of creation of the model
        into the newly generated "notes.txt" file within the 
        directory of the model. 

        Should only be called when the model is created.
        """
        self.write(f"Created: {self.model.timeOfCreation.strftime('%A, %d-%m-%Y %X')}\n\n")

    def write(self, text: str, mode: str = "a"):
        """
        A function to write something into the "notes.txt" file.

        Parameters
        ----------
        `text`: The string of text which to write into the "notes.txt" file.

        `mode`: Which writing mode to use in writing to the "notes.txt" file.
        \t      If mode = "a", `text` will be added to "notes.txt",
        \t      if mode = "w", `text` will completely overwrite "notes.txt".
        """
        with open(self.model.fullPath + "notes.txt", mode) as file: file.write(text)

    def load(self):
        """
        Function which loads the entire contents of the "notes.txt" file.

        Returns
        -------
        `file_contents`: An array containing all the lines of the "notes.txt" file.
        """
        with open(self.model.fullPath + "notes.txt", "r") as file:
            return file.readlines()
        
    def getConstants(self) -> dict:
        """
        Function which extracts the constants of the model from the "notes.txt" file.

        Returns
        -------
        'dictionary': A dictionary containing the base constants of the model.
        """
        lines = self.load()
        constants_index  = np.where(["Constants" in line for line in lines])[0][0]
        constants_string = lines[constants_index+1:constants_index+12]
        return {line[:line.find("=")-1]: str_convert(line[line.find("=")+2:line.find("[")]) for line in constants_string}

    def writeConstants(self):
        """
        Function which writes the constants of the model into the "notes.txt" file.
        Will overwrite already written constants in the "notes.txt" file.
        """
        constants_string = ("# Constants\n"
        f"log(M200) = {np.log10(self.model.M200/u.M_sun):.3e} [M_sun]\n"
        f"concentration = {self.model.conInput}\n"
        f"H0 = {self.model.H0.to(u.km/u.s/u.Mpc).value} [km / s / Mpc]\n"
        f"sigma/m = {self.model.scatteringParam.to(u.cm**2/u.g).value} [cm^2 / g]\n"
        f"halo Age = {self.model.haloAge.to(u.Gyr).value} [Gyr]\n"
        f"rhoMean = {self.model.NFW.rhoMean.to(u.M_sun/u.Mpc**3).value} [M_sun / Mpc^3]\n"
        f"mu = {self.model.NFW.mu}\n"
        f"phi = {self.model.NFW.phi}\n"
        f"chi_rho = {self.model.EoS.chi_rho}\n"
        f"chi_T = {self.model.EoS.chi_T}\n"
        f"chi_mu = {self.model.EoS.chi_mu}\n")

        constants_length = len(constants_string.split("\n"))-1
        
        if sum(["Constants" in line for line in self.load()]) == 0: 
            self.write(constants_string)
        else:
            lines = self.load()
            constants_index  = np.where(["Constants" in line for line in lines])[0][0]
            lines[constants_index:constants_index+constants_length] = [line + "\n" for line in constants_string.split("\n")[:-1]]
            self.write("".join(lines), mode="w")
    
    def getM200s_powers(self) -> np.ndarray[float]:
        """
        Function which extracts the log(M200)-values written in "notes.txt",
        which correspond the M200s used in generating "G-Grid.csv".

        Returns
        -------
        `M200s_powers`: An array of floats containing the extracted log(M200)-values.
        """
        lines = self.load()
        M200s_powers_line  = lines[np.where(["M200s-powers" in line for line in lines])[0][0] + 1] 
        M200s_powers       = np.array([float(M200_power) for M200_power in M200s_powers_line.split("=")[1].split(", ")])
        return M200s_powers
    
    def writeM200s_powers(self, M200s_powers: np.ndarray[float]):
        """
        Function which writes the inserted log(M200)-values into the "notes.txt" file.
        Will overwrite already written log(M200)-values in the "notes.txt" file.

        Parameters
        ----------
        `M200s_powers`: An array of floats representing the log(M200)-values
        \t              to be written into the "notes.txt" file.
        """
        M200s_powers_string = "# M200s-powers\nlog(M200s) [M_sun] = " + f"{list(M200s_powers)}"[1:-1]+"\n"
        
        if sum(["M200s-powers" in line for line in self.load()]) == 0: 
            self.write("\n"+M200s_powers_string)
        else:
            lines = self.load()
            M200s_powers_index  = np.where(["M200s-powers" in line for line in lines])[0][0]
            lines[M200s_powers_index:M200s_powers_index+2] = [line + "\n" for line in M200s_powers_string.split("\n")[:-1]]
            self.write("".join(lines)+"\n", mode="w")

    def writeProperties(self):
        """
        Function which writes the derived properties of the model into the "notes.txt" file.
        Will overwrite already written derived properties in the "notes.txt" file.
        """
        properties_string = ("# Derived Properties\n\n"
        "# HaloModel\n"
        f"h = {self.model.h.value}\n"
        f"rho_crit = {self.model.rho_crit/(u.M_sun/u.Mpc**3)} [M_sun / Mpc^3]\n"
        f"rho200 = {self.model.rho200/(u.M_sun/u.Mpc**3)} [M_sun / Mpc^3]\n"
        f"R200 = {self.model.R200/(u.Mpc)} [Mpc]\n"
        f"rs = {self.model.rs/(u.Mpc)} [Mpc]\n"
        f"V200 = {self.model.V200/(u.km/u.s)} [km / s]\n\n"
        "# NFW\n"
        f"logRhoMean = {self.model.NFW.logRhoMean} [M_sun / Mpc^3]\n"
        f"delta = {self.model.NFW.delta}\n"
        f"rho0 = {self.model.NFW.rho0/(u.M_sun/u.Mpc**3)} [M_sun / Mpc^3]\n"
        f"T200 = {self.model.NFW.T200/u.K} [K]\n\n"
        "# SIDM\n"
        f"r_trans = {self.model.SIDM.transRadius/u.Mpc} [Mpc]\n"
        f"rho_trans = {self.model.SIDM.transDens/(u.M_sun/u.Mpc**3)} [M_sun / Mpc^3]\n"
        f"M_trans = {self.model.SIDM.transMass/(u.M_sun)} [M_sun]\n"
        f"King radius = {self.model.SIDM.KingRadius/u.Mpc} [Mpc]\n"
        f"rho_central = {self.model.SIDM.centralDens/(u.M_sun/u.Mpc**3)} [M_sun / Mpc^3]\n")

        properties_length = len(properties_string.split("\n"))-1
        
        if sum(["Properties" in line for line in self.load()]) == 0: 
            self.write(properties_string)
        else:
            lines = self.load()
            properties_index  = np.where(["Properties" in line for line in lines])[0][0]
            lines[properties_index:properties_index+properties_length] = [line + "\n" for line in properties_string.split("\n")[:-1]]
            self.write("".join(lines), mode="w")

class Plotter:
    def __init__(self, haloModel: HaloModel):
        """
        NEEDS DOCUMENTATION
        """
        self.model = haloModel
        self.standardSetup()

    def standardSetup(self, setup: dict | str = 'default'):
        """
        Standard setup for what figures look like.

        Parameters
        ----------
        `setup`: Either a dictionary or string. If `setup` is a dictionary,
        \t       it will be parsed in `plt.rcParams.update()`. If `setup`
        \t       equals 'default', matplotlib will use its default setup. If
        \t       `setup` equals 'custom', will use the custom setup.
        """
        
        default_setup = plt.rcParams.copy()

        custom_setup = {"font.family":      "Times New Roman",
                     "font.serif":          "Times New Roman",
                     "font.size":           18,
                     "text.usetex":         True, 
                     "axes.grid":           True,
                     "xtick.minor.visible": True,
                     "ytick.minor.visible": True,
                     "xtick.direction":     "inout",
                     "ytick.direction":     "inout",
                     "xtick.top":           True,
                     "ytick.right":         True,
                     "savefig.format":      "pdf",
                     "lines.linewidth":     3.0,
                     "axes.facecolor":      "whitesmoke",
                     "xtick.major.size":    12.0,
                     "ytick.major.size":    12.0,
                     "xtick.minor.size":    6.0,
                     "ytick.minor.size":    6.0,
                     "xtick.major.width":   1.0,
                     "ytick.major.width":   1.0,
                     "xtick.minor.width":   1.0,
                     "ytick.minor.width":   1.0,
                     "axes.prop_cycle":     cycler(color=["#045275", "#089099", "#7CCBA2", "#FFCF66", "#F0746E", "#DC3977", "#7C1D6F"]),
                     "scatter.marker":      "h",
                     "figure.figsize":      [6*1.618033988749895,6],
                     "figure.dpi":          100,
                     "errorbar.capsize":    4,
                     "lines.marker":        "h",
                     "image.cmap":          "coolwarm",
                     "legend.loc":          (1.01,0),
                     "legend.edgecolor":    "k",
                     }
        
        if type(setup) is dict:
            plt.rcParams.update(setup)
        elif setup == 'custom':
            plt.rcParams.update(custom_setup)
        elif setup == 'default':
            plt.rcParams.update(default_setup)
        else:
            raise ValueError(f"Unknown input '{setup}' for the parameter 'setup'.")


    def colorbar(self, fig: Figure, ax: Axes, cmap: str, lims: tuple, label: str, width: float = 10) -> Axes:
        """
        Standard setup for colorbars.

        Parameters
        ----------
        `fig`: The matplotlib figure to create a colorbar into.

        `ax`: The matplotlib axis to create a colorbar for.

        `cmap`: The colormap to use.

        `lims`: String which indicates the limits of the colorbar.

        `label`: String, gives the y-label of the colorbar.

        `width`: Float, width of the colorbar.

        Returns
        -------
        `cbar_ax`: The matplotlib axis instance which points to the axis
        \t         in which the colorbar was created.
        """
        ax.tick_params(which='both', right = False) 
        
        cbar_ax = fig.add_axes([ax.get_position().xmax, ax.get_position().ymin, fig.subplotpars.wspace/width, ax.get_position().ymax - ax.get_position().ymin])
        norm = Normalize(vmin=lims[0], vmax=lims[1])
        ColorbarBase(cbar_ax, cmap=cmap, norm=norm)

        cbar_ax.minorticks_on()
        cbar_ax.yaxis.tick_right()
        cbar_ax.yaxis.set_label_position("right")
        cbar_ax.set_xticks([])
        cbar_ax.grid()
        cbar_ax.tick_params(which='both', direction='out') 

        cbar_ax.set_ylim(*lims)
        cbar_ax.set_ylabel(label)
        return cbar_ax

    def savefig(self, fig: Figure, name: str):
        """
        A function which saves a figure into the "Figures" folder inside the directory of the model.
        This function also closes the inserted figure.

        Parameters
        ----------
        `fig`: The matplotlib figure to save.

        `name`: The string which will be used at the file-name for the saved figure.
        """

        figures_folder = self.model.fullPath + "Figures/"
        if self.model.scatteringParam.value != 0: 
            figures_folder = self.model.fullPath + f"Figures/(sigma = {self.model.scatteringParam.value})/"
        
        os.makedirs(figures_folder, exist_ok=True)
        fig.savefig(figures_folder+name)
        plt.close(fig)

    def plotAll(self, cmap: str ='coolwarm'):
        """
        A function for convenience.

        Generates and saves all the figures which this class
        can generate, displaying all relevant data for all M200-values.
        """
        self.gasMassvsDMMass()
        self.Gvalues(cmap=cmap)
        self.Fvalues()
        self.densProfiles(cmap=cmap)
        self.nHIProfiles(cmap=cmap)
        self.NHIProfiles(cmap=cmap)
        self.recreateFig7(cmap=cmap)
        self.recreateFig6('all', cmap=cmap)

    def gasMassvsDMMass(self, M200s_powers: float | list | None = None, r_mask: tuple = (0, 1)):
        """
        Generates a figure displaying the calculated gass masses versus the used virial masses.
        Overlays this data on top of Figure 1 from Benitez-Llambay (2017). 
        
        Parameters
        ----------
        `M200s_powers`: A float, an array of floats, representing the log(M200)-values
        \t              and corresponding gas mass values to plot.
        \t              Alternatively, if set to None will create a figure for all 
        \t              M200-values used in generating "G-Grid.csv".
        
        `r_mask`: The with respect to R200 normalized radius range between which to calculate the gas masses.
        \t        `r_mask` must be a tuple (or list) of size 2, giving the beginning and endpoint of the integration range.
        """
        if M200s_powers is None:
            M200s_powers = self.model.notes.getM200s_powers()
         
        M_gass = self.model.densProfile.gasMassCalculator(M200s_powers, r_mask)

        fig = plt.figure(figsize=(10,10))

        try:
            paperFig = plt.imread(os.getcwd() + "/Figs from Paper/Benitez-Llambay (2017)/Figure 1/Figure 1.jpg")

            plt.imshow(paperFig, extent=[8, 12, 3, 11.4], aspect=(12-8)/(11.4-3))
        except FileNotFoundError:
            pass
        
        plt.plot(M200s_powers, np.log10(M_gass/u.M_sun), c="magenta", ls="--", marker="")

        plt.xlim(8, 12), plt.ylim(3, 11.4)

        plt.xticks(np.arange(8,12+1)), plt.yticks(np.arange(3,11+1))

        plt.xlabel(f"$\\log M_{{200}} \\left[\\mathrm{{M_\\odot}} \\right]$"), plt.ylabel(f"$\\log M_\\mathrm{{gas}} \\left[\\mathrm{{M_\\odot}}\\right]$")

        plt.grid()
        self.savefig(fig, "Gas-Mass vs DM-Mass")
    
    def Fvalues(self):
        """
        Generates a plot displaying the calculated F-values versus the log(rho)-values
        inside the generated "F-Grid.csv" file.
        """
        F_Grid      = np.loadtxt(self.model.fullPath + "F-Grid.csv", delimiter=",")
        logrhos, Fs = F_Grid[:,0], F_Grid[:,1]

        F_mask = Fs > 0 

        fig = plt.figure(figsize=(10,10))
        plt.plot(logrhos[F_mask], np.log10(Fs[F_mask]))

        lower_logrho, upper_logrho = 0.5*np.floor(logrhos[F_mask][0]/0.5), 0.5*np.ceil(logrhos[F_mask][-1]/0.5)
        lower_logF,   upper_logF   = 0.5*np.floor(np.log10(Fs[F_mask])[0]/0.5), 0.5*np.ceil(np.log10(Fs[F_mask])[-1]/0.5)

        plt.xlim(lower_logrho, upper_logrho), plt.ylim(lower_logF, upper_logF)
        plt.xticks(np.arange(lower_logrho, upper_logrho+2, 2)), plt.yticks(np.arange(lower_logF, upper_logF+0.5, 0.5))

        plt.xlabel("$\\log \\rho\\;\\left[\\mathrm{M_\\odot\\;Mpc^{-3}}\\right]$"), plt.ylabel("$\\log F \\left[\\mathrm{K}\\right]$")
        self.savefig(fig, "F-Values")

    def Gvalues(self, cmap: str = 'coolwarm'):
        """
        Generates a plot displaying the calculated G-values versus the
        with respect to R200 normalized radii used in the "G-Grid.csv" file
        for each M200.

        Parameters
        ----------
        `cmap`: The colormap to use in the plot.
        """
        M200s_powers = self.model.notes.getM200s_powers()
        G_Grid       = np.loadtxt(self.model.gridPath('G-Grid'), delimiter=",")
        r_norms      = G_Grid[:,0]

        fig = plt.figure(figsize=(10,10))
        fig.subplots_adjust(right=0.85)
        ax  = plt.gca()

        colors = plt.get_cmap(cmap, len(M200s_powers))
        for i in range(len(G_Grid[0])-1):
            Gs = G_Grid[:,i+1]

            color = colors(i/(len(G_Grid[0])-1))

            ax.plot(np.log10(r_norms), np.log10(Gs), c=color)


        self.colorbar(fig, ax, colors, (M200s_powers[0], M200s_powers[-1]), "$\\log M_{200}\\;\\left[\\mathrm{M_\\odot}\\right]$")

        lower_logr_norm, upper_logr_norm = 0.5*np.floor(np.log10(r_norms[0])/0.5)          , 0.5*np.ceil(np.log10(r_norms[-1])/0.5)
        lower_logG     , upper_logG      = 0.5*np.floor(np.log10(np.min(G_Grid[:,1:]))/0.5), 0.5*np.ceil(np.log10(np.max(G_Grid[:,1:]))/0.5)

        ax.set_xlim(lower_logr_norm, upper_logr_norm), ax.set_ylim(lower_logG, upper_logG)
        ax.set_xticks(np.arange(lower_logr_norm, upper_logr_norm+0.5, 0.5)), ax.set_yticks(np.arange(lower_logG, upper_logG+0.5, 0.5))

        ax.set_xlabel("$\\log r/R_{200}$"), ax.set_ylabel("$\\log G \\left[\\mathrm{K}\\right]$")
        self.savefig(fig, "G-Values")


    def densProfiles(self, cmap: str = 'coolwarm', r_mask: tuple = (0, np.inf)):
        """
        Generates a plot displaying the calculated gas density profiles versus the
        with respect to R200 normalized radii used in the "Density Profiles.csv" file
        for each M200.

        Parameters
        ----------
        `cmap`: The colormap to use in the plot.

        `r_mask`: The with respect to R200 normalized radius range between which to show the gas density profiles.
        \t        `r_mask` must be a tuple (or list) of size 2, giving the beginning and endpoint of the normalized radius range.
        """
        M200s_powers = self.model.notes.getM200s_powers()
        densProfiles = np.loadtxt(self.model.gridPath('Density Profiles'), delimiter=",")
        r_norms      = densProfiles[:,0]

        mask = (r_mask[0] <= r_norms) & (r_norms <= r_mask[1])

        fig = plt.figure(figsize=(10,10))
        fig.subplots_adjust(right=0.85)
        ax  = plt.gca()

        colors = plt.get_cmap(cmap, len(M200s_powers))
        for i in range(len(densProfiles[0])-1):
            rhos = densProfiles[:,i+1]

            color = colors(i/(len(densProfiles[0])-1))

            ax.plot(np.log10(r_norms[mask]), np.log10(rhos[mask]), c=color)

        ax.axhline(y=self.model.NFW.logRhoMean, c="k", ls="--")

        self.colorbar(fig, ax, colors, (M200s_powers[0], M200s_powers[-1]), "$\\log M_{200}\\;\\left[\\mathrm{M_\\odot}\\right]$")

        lower_logr_norm, upper_logr_norm = 0.5*np.floor(np.log10(r_norms[mask][0])/0.5)          , 0.5*np.ceil(np.log10(r_norms[mask][-1])/0.5)
        lower_logrho   , upper_logrho    = 0.5*np.floor(np.log10(np.min(densProfiles[:,1:]))/0.5), 0.5*np.ceil(np.log10(np.max(densProfiles[:,1:]))/0.5)

        ax.set_xlim(lower_logr_norm, upper_logr_norm), ax.set_ylim(lower_logrho, upper_logrho)
        ax.set_xticks(np.arange(lower_logr_norm, upper_logr_norm+0.5, 0.5)), ax.set_yticks(np.arange(lower_logrho, upper_logrho+1, 1))

        ax.set_xlabel("$\\log r/R_{200}$"), ax.set_ylabel("$\\log \\rho_\\mathrm{b} \\left[\\mathrm{M_\\odot\\;Mpc^{-3}}\\right]$")
        self.savefig(fig, "Density Profiles")


    def nHIProfiles(self, cmap: str = 'coolwarm', r_mask: tuple = (0, np.inf)):
        """
        Generates a plot displaying the calculated neutral hydrogen number densities versus the
        with respect to R200 normalized radii used in the "HI-Density Profiles.csv" file
        for each M200.

        Parameters
        ----------
        `cmap`: The colormap to use in the plot.

        `r_mask`: The with respect to R200 normalized radius range between which to show the density profiles.
        \t        `r_mask` must be a tuple (or list) of size 2, giving the beginning and endpoint of the normalized radius range.
        """
        M200s_powers   = self.model.notes.getM200s_powers()
        HIdensProfiles = np.loadtxt(self.model.gridPath('HI-Density Profiles'), delimiter=",")
        r_norms        = HIdensProfiles[:,0]
        nHIProfiles    = HIdensProfiles[:,1:]

        mask = (r_mask[0] <= r_norms) & (r_norms <= r_mask[1])

        fig = plt.figure(figsize=(10,10))
        fig.subplots_adjust(right=0.85)
        ax  = plt.gca()

        colors = plt.get_cmap(cmap, len(M200s_powers))
        for i in range(len(nHIProfiles[0])-1):
            nHIs = nHIProfiles[:,i]

            color = colors(i/(len(nHIProfiles[0])-1))

            ax.plot(np.log10(r_norms[mask]), np.log10(nHIs[mask]), c=color)

        self.colorbar(fig, ax, colors, (M200s_powers[0], M200s_powers[-1]), "$\\log M_{200}\\;\\left[\\mathrm{M_\\odot}\\right]$")

        lower_logr_norm, upper_logr_norm = 0.5*np.floor(np.log10(r_norms[mask][0])/0.5)                    , 0.5*np.ceil(np.log10(r_norms[mask][-1])/0.5)
        lower_logrho   , upper_logrho    = 0.5*np.floor(np.log10(np.min(nHIProfiles[:,1:]))/0.5), 0.5*np.ceil(np.log10(np.max(nHIProfiles[:,1:]))/0.5)

        ax.set_xlim(lower_logr_norm, upper_logr_norm), ax.set_ylim(lower_logrho, upper_logrho)
        ax.set_xticks(np.arange(lower_logr_norm, upper_logr_norm+0.5, 0.5)), ax.set_yticks(np.arange(lower_logrho, upper_logrho+1, 1))

        ax.set_xlabel("$\\log r/R_{200}$"), ax.set_ylabel("$\\log n_\\mathrm{HI} \\left[\\mathrm{cm^{-3}}\\right]$")
        self.savefig(fig, "HI-Density Profiles")

    def NHIProfiles(self, cmap: str = 'coolwarm', r_mask: tuple = (0, np.inf)):
        """
        Generates a plot displaying the calculated neutral hydrogen column densities versus the
        with respect to R200 normalized radii used in the "HI-Column Density Profiles.csv" file
        for each M200.

        Parameters
        ----------
        `cmap`: The colormap to use in the plot.

        `r_mask`: The with respect to R200 normalized radius range between which to show the column density profiles.
        \t        `r_mask` must be a tuple (or list) of size 2, giving the beginning and endpoint of the normalized radius range.
        """
        M200s_powers = self.model.notes.getM200s_powers()
        NHIProfiles  = np.loadtxt(self.model.gridPath('HI-Column Density Profiles'), delimiter=",")
        r_norms      = NHIProfiles[:,0]

        mask = (r_mask[0] <= r_norms) & (r_norms <= r_mask[1])

        fig = plt.figure(figsize=(10,10))
        fig.subplots_adjust(right=0.85)
        ax  = plt.gca()

        colors = plt.get_cmap(cmap, len(M200s_powers))
        for i in range(len(NHIProfiles[0])-1):
            rhos = NHIProfiles[:,i+1]

            color = colors(i/(len(NHIProfiles[0])-1))

            ax.plot(np.log10(r_norms[mask]), np.log10(rhos[mask]), c=color)

        self.colorbar(fig, ax, colors, (M200s_powers[0], M200s_powers[-1]), "$\\log M_{200}\\;\\left[\\mathrm{M_\\odot}\\right]$")

        lower_logr_norm, upper_logr_norm = 0.5*np.floor(np.log10(r_norms[mask][0])/0.5)         , 0.5*np.ceil(np.log10(r_norms[mask][-1])/0.5)
        lower_logrho   , upper_logrho    = 0.5*np.floor(np.log10(np.min(NHIProfiles[:,1:]))/0.5), 0.5*np.ceil(np.log10(np.max(NHIProfiles[:,1:]))/0.5)

        ax.set_xlim(lower_logr_norm, upper_logr_norm), ax.set_ylim(lower_logrho, upper_logrho)
        ax.set_xticks(np.arange(lower_logr_norm, upper_logr_norm+0.5, 0.5)), ax.set_yticks(np.arange(lower_logrho, upper_logrho+1, 1))

        ax.set_xlabel("$\\log r/R_{200}$"), ax.set_ylabel("$\\log N_\\mathrm{HI} \\left[\\mathrm{cm^{-2}}\\right]$")
        self.savefig(fig, "HI-Column Density Profiles")

    def recreateFig7(self, cmap: str = 'coolwarm'):
        """
        Generates a recreation of Figure 7 in Benitez-Llambay (2017)

        Parameters
        ----------
        `cmap`: The colormap to use in the plot.
        """
        M200s_powers = self.model.notes.getM200s_powers()
        
        yranges     = [[ -7, -0.5],
                       [ 9.5, 21.]]
        
        caxs        = [[2, -4.73, np.abs(2.037-2.228), np.abs(-4.73-(-1))],
                       [2, 13.82, np.abs(2.037-2.228), np.abs(13.82-20.4)]]

        cax_yranges = [[4.5, 9],
                       [8, 9.7]]

        cax_ticks   = [[4.5, 5, 6, 7, 8, 9],
                       [8.0, 8.5, 9.0, 9.5]]

        cax_ylabels = ["$\\log M_\\mathrm{gas} \\; \\left[ \\mathrm{M_\\odot}\\right]$",
                       "$\\log M_{200}         \\; \\left[ \\mathrm{M_\\odot}\\right]$"]

        ylabels     = ["$\\log n_\\mathrm{H}   \\; \\left[ \\mathrm{cm^{-3}}\\right]$",
                       "$\\log N_\\mathrm{HI}  \\; \\left[ \\mathrm{cm^{-2}}\\right]$"]

        yticks      = [np.arange(-7, -1+1, 1),
                       np.arange(10, 20+2, 2) ]

        fig, axs = plt.subplots(2, 1, figsize=(6,10), sharex=True)
        fig.subplots_adjust(left=0.2, hspace=0.1, right=0.85)

        colors = plt.get_cmap(cmap, len(M200s_powers))
        
        NHIs_Grid = np.loadtxt(self.model.gridPath('HI-Column Density Profiles'), delimiter=",")
        dens_Grid = np.loadtxt(self.model.gridPath('Density Profiles'), delimiter=",")

        radii = NHIs_Grid[:,0]
        NHIs  = NHIs_Grid[:,1:] / u.cm**2
        nHs   = (dens_Grid[:,1:] * (u.M_sun/u.Mpc**3)/c.m_p).to(1/u.cm**3)

        r200_index = np.argmin(np.abs(radii - 1))

        NHI200s = NHIs[r200_index]
        nH200s  = nHs[r200_index]

        Mgass  = self.model.densProfile.gasMassCalculator()

        R200s  = self.model.calculateR200(10**M200s_powers * self.model.M200.unit).to(u.kpc)
        for i in range(len(axs)):
            ax = axs[i]
            cax = ax.inset_axes(caxs[i], transform=ax.transData)
            #cax = fig.add_axes([ax.get_position().xmax, ax.get_position().ymin, fig.subplotpars.wspace/10, ax.get_position().ymax - ax.get_position().ymin])
            # cax.yaxis.set_label_position("right"), cax.yaxis.tick_right()
            cax.set_ylim(*cax_yranges[i]), cax.set_yticks(cax_ticks[i])
            cax.xaxis.set_ticklabels([]), cax.set_xticks([])
            cax.set_ylabel(cax_ylabels[i])
            cax.grid()
            cax.tick_params(which='both', right=False) 
            cax.tick_params(which='both', direction="out") 

            ax.axhline(y=self.model.NFW.logRhoMean+ np.log10(1*(u.M_sun/u.Mpc**3)/c.m_p/(1/u.cm**3)), c="k", ls="-.", lw=1)

            for j, M200s_pow in enumerate(M200s_powers):
                color = colors(j/len(M200s_powers))

                ydata = np.log10([nHs, NHIs][i][:,j].value)
                xdata = np.log10(radii*R200s[j]/u.kpc)

                ax.plot(xdata, ydata, c=color, lw=1)

                if i == 0: 
                    cax.axhline(y = np.log10(Mgass[j]/u.M_sun), c=color, lw=1)
                if i == 1: cax.axhline(y = M200s_pow, c=color, lw=1)

            ax.plot(np.log10(R200s/u.kpc), np.log10([nH200s, NHI200s][i].value), c="k", ls="--", lw=1)

            ax.set_xlim(-0.5, 2.4), ax.set_ylim(*yranges[i])
            ax.set_ylabel(ylabels[i])
            ax.set_yticks(yticks[i])
        axs[1].set_xlabel("$\\log r \\; \\left[\\mathrm{kpc}\\right]$")
        axs[1].set_xlabel("$\\log r \\; \\left[\\mathrm{kpc}\\right]$")
        axs[1].set_xticks(np.arange(-0.5, 2.0+0.5, 0.5))

        self.savefig(fig, "Fig 7 recreation")

    def recreateFig6(self, M200s_powers: float | np.ndarray | None = None, cmap: str ='coolwarm', r_mask: tuple =(0, np.inf), displayFigs: bool = False):
        """
        Generates a recreation of Figure 6 in Benitez-Llambay (2017)

        Can either be generated for all used virial masses or a selection of virial masses,
        and can be customized to a desired normalized radius range.

        Parameters
        ----------
        `M200s_powers`: A float, an array of floats, representing the log(M200)-values to plot.
        \t              Alternatively, if set to None will create a figure for all 
        \t              M200-values used in generating "G-Grid.csv".

        `cmap`: The colormap to use in the plot.

        `r_mask`: The with respect to R200 normalized radius range between which to recreate Figure 6.
        \t        `r_mask` must be a tuple (or list) of size 2, giving the beginning and endpoint of the normalized radius range.
        
        `displayFigs`: A boolean which determines wether to display the Figure 6 panels from Benitez-Llambay et al. (2017)
        \t             in the background of the panels. Note: will limit the x and y limits.
        """
        allM200s_powers = self.model.notes.getM200s_powers()
        
        if M200s_powers is None:
            M200s_powers = [np.log10(self.model.M200.value)]
        
        if type(M200s_powers) not in [list, np.ndarray] and M200s_powers != 'all':
            M200s_powers = [M200s_powers]

        if M200s_powers == 'all':
            M200s_powers = allM200s_powers
        
        if np.any(~np.isin(M200s_powers, allM200s_powers)):
            raise KeyError("One or more inputted M200s do not have the required data.")

        indices = np.searchsorted(allM200s_powers, M200s_powers)

        colors = plt.get_cmap(cmap, len(M200s_powers))

        lw = 3
        #if len(M200s_powers) < 10: lw = 3


        fig, axs = plt.subplots(2,2, figsize=(10,10), sharex=True)
        fig.subplots_adjust(hspace=0.1, wspace=0.1, right=0.85)

        ax1, ax2, ax3, ax4 = axs.flatten()

        densGrid = np.loadtxt(self.model.gridPath("Density Profiles"),           delimiter=",")
        TGrid    = np.loadtxt(self.model.gridPath("T Profiles"),                 delimiter=",")
        nHIGrid  = np.loadtxt(self.model.gridPath("HI-Density Profiles"),        delimiter=",")
        NHIGrid  = np.loadtxt(self.model.gridPath("HI-Column Density Profiles"), delimiter=",")
        
        radii    = densGrid[:,0]

        r_mask = (r_mask[0] <= radii) & (radii <= r_mask[1])

        logradii = np.log10(radii)[r_mask]

        temp_model = HaloModel(directory=False)
        temp_model.constantsSetter(self.model.notes.getConstants())
        for index in indices:
            temp_model.M200 = 10**allM200s_powers[index]
            
            color = colors(index/len(allM200s_powers))
            color = "C2"

            logAcc_data = np.log10(temp_model.SIDM.gravitationalAcceleration(radii*temp_model.R200)/(u.cm/u.s**2))[r_mask]
            lognH_data  = np.log10(densGrid[:,index+1] * (u.M_sun/u.Mpc**3)/c.m_p/(1/u.cm**3))[r_mask]
            lognHI_data = np.log10(nHIGrid[:,index+1])[r_mask]
            logT_data   = np.log10(TGrid[:,index+1])[r_mask]
            logNHI_data = np.log10(NHIGrid[:,index+1])[r_mask]

            ax1.plot(logradii, logAcc_data, c=color, lw=lw, marker='')
            ax2.plot(logradii, lognH_data , c=color, lw=lw, marker='')
            ax2.plot(logradii, lognHI_data, c=color, lw=lw, ls="--", marker='')
            ax3.plot(logradii, logT_data  , c=color, lw=lw, marker='')
            ax4.plot(logradii, logNHI_data, c=color, lw=lw, marker='')

            if displayFigs:
                ax2.arrow(x=1.2-1.5, y=np.log10((self.model.NFW.rhoMean/c.m_p).to(1/u.cm**3).value), dx=0.25, dy=0, color=color, head_length=0.04, head_width=0.2, length_includes_head=True)
                ax3.arrow(x=-0.15-1.5, y=logT_data[np.argmin(np.abs(radii - 1))], dx=-0.25, dy=0, color=color, head_length=0.04, head_width=0.025, length_includes_head=True)


        ax1.set_ylabel("$\\log \\frac{GM}{r^2}   \\; \\left[ \\mathrm{cm\\;s^{-2}} \\right]$")
        ax2.set_ylabel("$\\log n_\\mathrm{H}     \\; \\left[ \\mathrm{cm^{-3}}     \\right]$")
        ax3.set_ylabel("$\\log T                 \\; \\left[ \\mathrm{K}           \\right]$")
        ax4.set_ylabel("$\\log N_\\mathrm{HI}    \\; \\left[ \\mathrm{cm^{-2}}     \\right]$")

        ax3.set_xlabel("$\\log \\tilde{r}$"), ax4.set_xlabel("$\\log \\tilde{r}$")
        
        ax2.yaxis.set_label_position("right"), ax2.yaxis.tick_right()
        ax4.yaxis.set_label_position("right"), ax4.yaxis.tick_right()

        ax1.set_xlim(np.floor(logradii[0]), np.ceil(logradii[-1]))
        ax1.set_xticks(np.arange(ax1.get_xlim()[0], ax1.get_xlim()[1]+1, 1))

        if displayFigs:

            yranges =  [[-10.4, -8.5000],
                        [ -8.5, -1.0000],
                        [  4.0,  4.6975],
                        [12.76, 21.0000]]

            xrange   = np.array([-2.0, 0.0]) + 0.025

            yTickSteps = [0.5, 1, 0.2, 1] 


            for i in range(4):
                ax = axs.flatten()[i]
                figFromPaper = plt.imread(os.getcwd() + f"/Figs from Paper/Benitez-Llambay (2017)/Figure 6/Figure 6.{int(i+1)}.png")
                ax.imshow(figFromPaper, extent=[xrange[0], xrange[1], yranges[i][0], yranges[i][1]], aspect=np.abs(xrange[1]-xrange[0])/np.abs(yranges[i][1]-yranges[i][0]))
                
                yticks = np.arange(yranges[i][1], yranges[i][0]-yTickSteps[i], -yTickSteps[i])
                if i == 2: yticks = np.arange(yranges[2][0], yranges[2][1], yTickSteps[i])
    
                ax.set_yticks(np.sort(yticks)), ax.set_xticks(np.arange(-1.5, 0+0.5, 0.5))
                ax.set_xlim(*xrange), ax.set_ylim(*yranges[i])

                ax.grid()

        # cbar_ax = fig.add_axes([ax1.get_position().xmin, ax1.get_position().ymax, ax2.get_position().xmax - ax1.get_position().xmin, fig.subplotpars.hspace/5])
        # ax1.tick_params(which='both', top = False) 
        # ax2.tick_params(which='both', top = False) 

        # norm = Normalize(vmin=allM200s_powers[0], vmax=allM200s_powers[-1])
        # ColorbarBase(cbar_ax, cmap=colors, norm=norm, orientation='horizontal')

        # cbar_ax.minorticks_on()
        # cbar_ax.xaxis.tick_top()
        # cbar_ax.xaxis.set_label_position("top")
        # cbar_ax.set_yticks([])
        # cbar_ax.grid()
        # cbar_ax.tick_params(which='both', direction='out') 

        # cbar_ax.set_xlim(allM200s_powers[0], allM200s_powers[-1])
        # cbar_ax.set_xlabel("$\\log M_{200}\\;\\left[\\mathrm{M_\\odot}\\right]$", labelpad=10)
    
        self.savefig(fig, "Fig 6 recreation")
