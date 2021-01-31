'''
DEB pipeline as in DEB-IBM (see code from Netlogo below)

Changes from code from netlogo :
 - __ instead of ^ in variable names
'''
import json
import numpy as np

from lib.stor.paths import get_parent_dir

'''
; ==========================================================================================================================================
; ========================== DEFINITION OF PARAMETERS AND STATE VARIABLES ==================================================================
; ==========================================================================================================================================

; global parameters: are accessible for patches and turtles
globals[
  U_E^0    ; t L^2, initial reserves of the embryos at the start of the simulation
  f        ; - , scal functional response
  L_0      ; cm, initial structural volume
]
; ------------------------------------------------------------------------------------------------------------------------------------------
; parameters for the environment: here only prey density

patches-own[
  X        ; # / cm^2, prey density
  d_X      ; change of prey density in time
]
; ------------------------------------------------------------------------------------------------------------------------------------------

; definition of parameters for the individuals:
; the notation follows the DEBtool-notation as far as possible
; deviation: rates are indicated with "_rate" rather than a dot
; each individual(turtle) in the model has the following parameters
turtles-own[
  ; - - - - - - - - - - - - - - - STATE VARIABLES - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  L           ; cm, structural length
  dL          ; change of structural length in time
  U_H         ; t L^2, scal maturity
  dU_H        ; change of scal maturity in time
  U_E         ; t L^2, scal reserves
  dU_E        ; change of scal reserves in time
  e_scaled    ; - , scal reserves per unit of structure
  U_R         ; t L^2, scal energy in reproduction buffer (not standard DEB)
  dU_R        ; change of energy in reproduction buffer (reproduction rate)

  ; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ; - - - - - - - - - - - - - - - FLUXES (used by several submodels) - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  S_A         ; assimilation flux
  S_C         ; mobilisation flux

  ; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ; - - - - - - - - - - - - - - - EMBRYO (we use different state variable to not affect the state variable of the mother) - - - - - - - - --
  e_scaled_embryo
  e_ref
  U_E_embryo
  S_C_embryo
  U_H_embryo
  L_embryo
  dU_E_embryo
  dU_H_embryo
  dL_embryo
  ; parameters used to calculate the costs for an egg / initial reserves
  lower-bound ; lower boundary for shooting method
  upper-bound ; upper boundary for shooting method
  estimation  ; estimated value for the costs for an egg / initial reserve
  lay-egg?    ; parameter needed to hand over if an egg can be laid
  offspring-count ; with this parameter, the reproduction rate per turtle is shown on the interface
  sim         ; this keeps track of how many times the calc-egg-size loop is run

  ; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ; - - - - - - - - - - - - - - - STANDARD DEB PARAMETERS (with dimension and name) - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  g           ; - , energy investment ratio
  v_rate      ; cm /t , energy conductance (velocity)
  kap         ; - , allocation fraction to soma
  kap_R       ; - , reproduction efficiency
  k_M_rate    ; 1/t, somatic maintenance rate coefficient
  k_J_rate    ; 1/t, maturity maintenance rate coefficient
  U_H^b       ; t L^2, scal maturity at birth
  U_H^p       ; t L^2, scal maturity at puberty
  ; parameter that is used to randomize the input parameters
  scatter-multiplier

  ; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ; - - - - - - - - - - - - - - - PREY DYNAMICS (only relevant if prey-dynamics are set to logistic) - - - - - - - - - - - - - - - - - - -

  J_XAm_rate  ; # / (cm^2 t), surface-area-specific maximum ingestion rate
  K           ; # / cm^2, (half) saturation coefficient

  ; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ; - - - - - - - - - - - - - - - AGEING -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  q_acceleration  ; - , ageing acceleration
  dq_acceleration ; change of ageing acceleration in time
  h_rate          ; - , hazard rate
  dh_rate         ; change of hazard rate in time
  age-day         ; each turtle has a random whole number between 0 and timestep if the mod of ticks = the age day of a turtle is will check to see if it dies
                  ; based on the ageing submodel. This is because mortality probabilities are per day, and timesteps are smaller
'''

'''
; ==========================================================================================================================================
; ========================== SETUP PROCEDURE: SETTING INITIAL CONDITIONS ===================================================================
; ==========================================================================================================================================

to setup
 ;; (for this model to work with NetLogo's new plotting features,
  ;; __clear-all-and-reset-ticks should be replaced with clear-all at
  ;; the beginning of your setup procedure and reset-ticks at the end
  ;; of the procedure.)
  __clear-all-and-reset-ticks

 if add_my_pet? = "on"
 [convert-parameters]

 set L_0 .00001   ; set initial length to some very small value (embryos start off as nearly all reserves)

 crt 10                   ; 10 turtles are created in the beginning
 ask  turtles  [
  individual-variability  ; first their individual variability in the parameter is set
  calc-embryo-reserve-investment     ; then the initial energy is calculated for each
 ]

 ask patches [ set X J_XAm_rate_int /   F_m ]; set initial value of prey to their carrying capacity
end
'''

'''
; ==========================================================================================================================================
; ========================== GO PROCEDURE: RUNNING THE MODEL ===============================================================================
; ==========================================================================================================================================
; the go statement below is the order in which all procedures are run each timestep

to go
   ask turtles
  [
    calc-dU_E                       ; first all individuals calculate the change in their state variables based on the current conditions
    calc-dU_H
    calc-dU_R
    calc-dL
  ]
  if aging = "on"                   ; if the ageing submodel is turned on, the change in damage inducing compound and damage are calculated
  [
    ask turtles
    [
      calc-dq_acceleration
      calc-dh_rate
    ]
  ]
  if food-dynamics = "logistic"     ; if prey dynamics are set to "logistic" the change in prey density is calculated
  [ask patches [calc-d_X]]

   update                           ; the the state variables of the individuals and prey are updated based on the delta value

  ask turtles with [U_H >= U_H^p] ; mature individual check if they have enough energy in their reproduction buffer to repdroduce
  [
    calc-lay-eggs
    if lay-egg? = 1
      [
        calc-embryo-reserve-investment         ; if so, they calculate how much energy to invest in an embryo
        lay-eggs                    ; and they produce one offspring
      ]
  ]
  tick
  do-plots                          ; then the plots are updated
  if count turtles = 0 [stop]

end

'''

'''
; ==========================================================================================================================================
; ========================== SUBMODELS =====================================================================================================
; ==========================================================================================================================================

; ---------------- conversion of parameters: from add_my_pet to standard DEB parameters ----------------------------------------------------

to convert-parameters
  let p_am p_m * zoom / kap_int
  set U_H^b_int E_H^b / p_am
  set U_H^p_int E_H^p / p_am
  set k_M_rate_int p_m / E_G
  set g_int (E_G * v_rate_int / p_am) / kap_int
end

; ------------------------------------------------------------------------------------------------------------------------------------------
; ------------------------ INDIVIDUAL VARIABILITY ------------------------------------------------------------------------------------------
; ------------------------------------------------------------------------------------------------------------------------------------------

to individual-variability
  ; individuals vary in their DEB paramters on a normal distribution with a mean on the input paramater and a coefficent of variation equal to the cv
  ; set cv to 0 for no variation
  set scatter-multiplier e ^ (random-normal 0 cv)
  set J_XAm_rate   J_XAm_rate_int * scatter-multiplier
  set g g_int / scatter-multiplier
  set U_H^b U_H^b_int / scatter-multiplier ;
  set U_H^p U_H^p_int / scatter-multiplier ;

  set v_rate v_rate_int
  set kap kap_int
  set kap_R kap_R_int
  set k_M_rate k_M_rate_int
  set k_J_rate k_J_rate_int
  set K  J_XAm_rate /   F_m
end
'''

'''
; ------------------------------------------------------------------------------------------------------------------------------------------
; ----------------- RESERVE DYNAMICS -------------------------------------------------------------------------------------------------------
; ------------------------------------------------------------------------------------------------------------------------------------------
; change in reserves: determined by the difference between assimilation (S_A) and mobilization (S_C) fluxes
; when food-dynamics are constant f = the value of f_scaled set in the user interface
; if food is set to  "logistic" f depends on prey density and the half-saturation coefficient (K)
; for embryos f = 0 because they do not feed exogenously

to calc-dU_E

  if food-dynamics = "constant"
  [ ifelse U_H <= U_H^b
    [set f 0]
    [set f f_scaled]
  ]
  if food-dynamics = "logistic"
  [ ifelse U_H <= U_H^b
    [set f 0]
    [set f X / (K + X)]
  ]
  set e_scaled v_rate * (U_E / L ^ 3)
  set S_C L ^ 2 * (g * e_scaled / (g + e_scaled)) * (1 + (L / (g * (V_rate / ( g * K_M_rate)))))

  set S_A  f * L ^ 2 ;

  set dU_E (S_A - S_C )
end
; ------------------------------------------------------------------------------------------------------------------------------------------
; ----------------- MATURITY AND REPRODUCTION  ---------------------------------------------------------------------------------------------
; ------------------------------------------------------------------------------------------------------------------------------------------
; change in maturity is calculated (for immature individuals only)

to calc-dU_H

  ifelse U_H < U_H^p ; they only invest into maturity until they reach puberty
    [set dU_H ((1 - kap) * S_C - k_J_rate * U_H) ]
    [set dU_H 0]
end

; the following procedure calculates change in reprobuffer if mature
to calc-dU_R
  if U_H >= U_H^p
    [set dU_R  ((1 - kap) * S_C - k_J_rate * U_H^p) ]
end

; ------------------------------------------------------------------------------------------------------------------------------------------
; ----------------- DYNAMICS OF STRUCTURAL LENGHT-------------------------------------------------------------------------------------------
; ------------------------------------------------------------------------------------------------------------------------------------------
; the following procedure calculates change in structural length, if growth in negative the individual does not have enough energy to pay somatic maintenance and the starvation submodel is run
; where growth is set to 0 and individuals divirt enough energy from development (for juveniles) or reprodution (for adults) to pay maintenance costs
to calc-dL

  set dL   ((1 / 3) * (((V_rate /( g * L ^ 2 )) * S_C) - k_M_rate * L))

  if e_scaled < L / (V_rate / ( g * K_M_rate))  ; if growth is negative use starvation strategy 3 from the DEB book
    [
      set dl 0
      ifelse U_H < U_H^p
       [set dU_H (1 - kap) * e_scaled * L ^ 2 - K_J_rate * U_H^p - kap * L ^ 2 * ( L / (V_rate / ( g * K_M_rate)) - e_scaled)]
       [ set dU_R  (1 - kap) * e_scaled * L ^ 2 - K_J_rate * U_H^p - kap * L ^ 2 * ( L / (V_rate / ( g * K_M_rate)) - e_scaled)]
      set dU_E  S_A - e_scaled * L ^ 2
   ifelse U_H < U_H^p

 [  if dU_H < 0 [die]]

      [if U_R < 0 [die]]
    ]

end
'''

'''
;------------------------------------------------------------------------------------------------------------------------------------------
;---------- CHECK IF POSSIBLE TO LAY EGGS -------------------------------------------------------------------------------------------------
;------------------------------------------------------------------------------------------------------------------------------------------
; in the following, individuals determine if they have enough energy in their repro buffer to reproduce by creating an embryo with initial reserves set to the energy
; currently in their repro buffer * kap_R (conversion efficiancy of  reprobuffer to embryo) if the individual has enough energy to produce an offspring which will reach
; maturity and have a reserve density greater than the mothers when it hatches "lay-egg?" is set to one which will trigger the reproduction procedures "calc-egg-size" and "lay-eggs"
to calc-lay-eggs
  set L_embryo  L_0
  set U_E_embryo U_R * kap_R
  set U_H_embryo  0

  loop [
    set e_scaled_embryo v_rate * (U_E_embryo / L_embryo  ^ 3)
    set S_C_embryo L_embryo  ^ 2 * (g * e_scaled_embryo / (g + e_scaled_embryo)) * (1 + (L_embryo  / (g * (V_rate / ( g * K_M_rate)))))

    set dU_E_embryo  ( -1 * S_C_embryo )
    set dU_H_embryo  ((1 - kap) * S_C_embryo - k_J_rate * U_H_embryo )
    set dL_embryo  ((1 / 3) * (((V_rate /( g * L_embryo  ^ 2 )) * S_C_embryo) - k_M_rate * L_embryo ))

    set  U_E_embryo  U_E_embryo +  dU_E_embryo  / timestep
    set  U_H_embryo  U_H_embryo  +  dU_H_embryo   / timestep
    set  L_embryo    L_embryo  +  dL_embryo   / timestep

    if U_H_embryo  > U_H^b  [ set lay-egg? 1 stop]
    if e_scaled_embryo < e_scaled [stop]
    ]
end


;-------------------------------------------------------------------------------------------------------------------------------------------
;--------- LAY EGGS ------------------------------------------------------------------------------------------------------------------------
;-------------------------------------------------------------------------------------------------------------------------------------------
;the following procedure is run for mature individuals which have enough energy to reproduce
; they create 1 offspring and give it the following state variables and DEB parameters
;the initial reserves is set to the value determined by the bisection method in "calc_egg_size"

to lay-eggs
  hatch 1
    [
      ;the following code give offspring varibility in their DEB paramters on a normal distribution with a mean on the input paramater and a coefficent of variation equal to the cv
      ; set cv to 0 for no variation

      set scatter-multiplier e ^ (random-normal 0 cv)
      set J_XAm_rate   J_XAm_rate_int * scatter-multiplier
      set g g_int / scatter-multiplier
      set U_H^b    U_H^b_int / scatter-multiplier
      set U_H^p    U_H^p_int / scatter-multiplier

      set v_rate v_rate_int

      set kap kap_int
      set kap_R kap_R_int
      set k_M_rate k_M_rate_int
      set k_J_rate k_J_rate_int
      set  K J_XAm_rate /   F_m

      set L L_0
      set U_E estimation
      set U_H 0
      set U_R 0
      set dU_R  0
      set h_rate 0
      set dh_rate 0
      set q_acceleration 0
      set dq_acceleration 0
      set lay-egg? 0
      set age-day random timestep
    ]
  set lay-egg? 0
  set U_R U_R - estimation
end

'''

'''
; ------------------------------------------------------------------------------------------------------------------------------------------
; ------------------------ INITIAL ENERGY --------------------------------------------------------------------------------------------------
; ------------------------------------------------------------------------------------------------------------------------------------------
; calculate the initial energy of the first individuals using a bisection method

to calc-embryo-reserve-investment
  set lower-bound 0
  ifelse ticks = 0
  [set upper-bound 100000]
  [set upper-bound U_R * kap_R]
  set sim 0

  loop[
    set sim sim + 1

    set estimation .5 * (lower-bound + upper-bound)
    set L_embryo  L_0
    set U_E_embryo estimation
    set U_H_embryo  0
    set e_scaled_embryo v_rate * (U_E_embryo / L_embryo  ^ 3)

    ifelse ticks = 0[set e_ref 1][set e_ref e_scaled]  ; e_ref now determines which e_scaled_embryo to calculate: 1 for ticks = 0 (in the setup procedure), e_scaled otherwise

    while [U_H_embryo  < U_H^b and e_scaled_embryo > e_ref ]
     [ set e_scaled_embryo v_rate * (U_E_embryo / L_embryo  ^ 3)
        set S_C_embryo L_embryo  ^ 2 * (g * e_scaled_embryo / (g + e_scaled_embryo)) * (1 + (L_embryo  / (g * (v_rate / ( g * k_M_rate)))))

        set dU_E_embryo  ( -1 * S_C_embryo )
        set dU_H_embryo  ((1 - kap) * S_C_embryo - k_J_rate * U_H_embryo  )
        set dL_embryo   ((1 / 3) * (((V_rate /( g * L_embryo  ^ 2 )) * S_C_embryo) - k_M_rate * L_embryo ))

        set  U_E_embryo  U_E_embryo +  dU_E_embryo    / (timestep )
        set  U_H_embryo   U_H_embryo  +  dU_H_embryo   / (timestep )
        set  L_embryo   L_embryo  +  dL_embryo    / (timestep )
      ]

    if e_scaled_embryo <  .05 +  e_ref and e_scaled_embryo > -.05 + e_ref and U_H_embryo  >= U_H^b  [

      ifelse ticks = 0 ;
      [set U_E^0 estimation
        set L L_0
        set U_E U_E^0
        set U_H 0
        set U_R 0
        set dU_R  0

        set age-day random timestep
        stop
      ][stop]]

    ifelse U_H_embryo  > U_H^b
      [ set upper-bound estimation ]
      [ set lower-bound estimation ]
    if sim > 200 [user-message ("Embryo submodel did not converge. Timestep may need to be smaller.") stop]
    ;if the timestep is too big relative to the speed of growth of species this will no converge
  ]
end
'''

'''

; ------------------------------------------------------------------------------------------------------------------------------------------
; ----------------- LOGISTIC PREY ----------------------------------------------------------------------------------------------------------
; ------------------------------------------------------------------------------------------------------------------------------------------
 ;the following procedure calculates change in prey density this procedure is only run when prey dynamics are set to "logistic" in the user interface

to calc-d_X
   set d_X (r_X) * X * (1 - (X / K_X))   - sum [ S_A * J_XAm_rate   ] of turtles-here / volume
end

; ------------------------------------------------------------------------------------------------------------------------------------------
; ----------------- AGEING -----------------------------------------------------------------------------------------------------------------
; ------------------------------------------------------------------------------------------------------------------------------------------
; the following procedure calculates the change in damage enducing compounds of an individual

to calc-dq_acceleration
  set dq_acceleration (q_acceleration * (L ^ 3 / (v_rate / ( g * k_M_rate)) ^ 3) * sG + H_a) * e_scaled * (( v_rate / L) - ((3 / L)*  dL)) - ((3 / L ) * dL) * q_acceleration
end

; the following procedure calculates the change in damage in the individual
to calc-dh_rate
  set dh_rate q_acceleration - ((3 / L) * dL) * h_rate
end

; ------------------------------------------------------------------------------------------------------------------------------------------
; ----------------- UPDATE -----------------------------------------------------------------------------------------------------------------
; ------------------------------------------------------------------------------------------------------------------------------------------

to update
; individuals update their state variables based on the calc_state variable proccesses
  ask turtles
  [
    set U_E U_E + dU_E / timestep
    set U_H U_H + dU_H / timestep
    set U_R U_R + dU_R    / timestep
    set L L + dL    / timestep
    if U_H > U_H^b
    [ set q_acceleration q_acceleration + dq_acceleration  / timestep
      set h_rate h_rate + dh_rate  / timestep
    ]

   if aging = "on" [if ticks mod timestep = age-day [if random-float 1 < h_rate [die]] ] ;ageing related mortality
   if aging = "off" [if ticks mod timestep = age-day [if random-float 1 < background-mortality [die]] ]
 ]
  if food-dynamics = "logistic"[ ask patches [ set X X + d_X / timestep]]
end
'''

'''
from https://www.bio.vu.nl/thb/deb/deblab/add_my_pet/entries_web/Aedes_aegypti/Aedes_aegypti_par.html

T_A 	8000 	    K	            Arrhenius temperature
p_Am 	227.109 	J/d.cm^2	    {p_Am}, spec assimilation flux => 2.27 J/d.mm^2
F_m 	6.5 	    l/d.cm^2	    {F_m}, max spec searching rate
kap_X 	0.8 	    -	            digestion efficiency of food to reserve
kap_P 	0.18 	    -	            faecation efficiency of food to faeces
v 	    0.018172 	cm/d	        energy conductance
kap 	0.67296 	-	            allocation fraction to soma
kap_R 	0.95 	    -	            reproduction efficiency
kap_V 	0.99148 	-	            conversion efficient E -> V -> E
p_M 	107.873 	J/d.cm^3	    [p_M], vol-spec somatic maint
p_T 	0 	        J/d.cm^2	    {p_T}, surf-spec somatic maint
k_J 	0.002 	    1/d	            maturity maint rate coefficient
E_G 	4400 	    J/cm^3	        [E_G], spec cost for structure
E_Hb 	0.009619 	J	            maturity at birth
s_j 	0.999 	    -	            reprod buffer/structure at pupation as fraction of max
E_He 	0.05218 	J	            maturity at emergence
h_a 	0.004218 	1/d^2	        Weibull aging acceleration
s_G 	0.0001 	    -	            Gompertz stress coefficient'''

default_deb_path = f'{get_parent_dir()}/lib/sim/deb_drosophila.csv'


class DEB:
    def __init__(self, species='aedes', steps_per_day=1, cv=0, aging=False, print_stage_change=False):
        self.print_stage_change = print_stage_change
        # My flags
        self.embryo = True
        self.larva = False
        self.puppa = False
        self.imago = False
        self.alive = True

        # My params
        # self.d = 1  # in g/cm**3, same for structure and reserve
        self.tick_counter = 0

        # Input params
        self.steps_per_day = steps_per_day
        self.cv = cv
        self.aging = aging
        self.species = species

        # Global parameters
        self.U_E__0 = None
        self.f = None
        self.L_0 = None

        # parameters for the environment: here only prey density
        self.X = None
        self.d_X = None

        # Individual parameters
        self.L = None
        self.dL = None
        self.U_H = None
        self.dU_H = None
        self.U_E = None
        self.dU_E = None
        self.e_scaled = None
        self.U_R = None
        self.dU_R = None
        self.U_V = None

        # Fluxes
        self.S_A = None
        self.S_C = None

        # EMBRYO
        self.e_ref = None
        self.U_E_embryo = None
        self.S_C_embryo = None
        self.U_H_embryo = None
        self.L_embryo = None
        self.dU_E_embryo = None
        self.dU_H_embryo = None
        self.dL_embryo = None

        #   parameters used to calculate the costs for an egg / initial reserves
        self.lower_bound = None
        self.upper_bound = None
        self.estimation = None
        self.lay_egg = None
        self.offspring_count = None
        self.sim = None

        # STANDARD DEB PARAMETERS
        self.g = None
        self.v_rate = None
        self.kap = None
        self.kap_R = None
        self.k_M_rate = None
        self.k_J_rate = None
        self.U_H__b = None
        # self.U_H__p = None
        self.U_H__p = 0.00001
        self.U_H__e = None
        #  parameter that is used to randomize the input parameters
        self.scatter_multiplier = None

        # PREY DYNAMICS
        # given from netlogo interface
        self.J_XAm_rate_int = 1
        self.F_m = 1
        self.r_X = 1
        self.K_X = 1
        self.volume = 1
        self.f_scaled = 1

        # not given from netlogo interface
        self.J_XAm_rate = None
        self.K = None

        # AGING
        # given from netlogo interface. Overriden by self.aedes()
        self.h_a = 4.105E-4
        self.sG = -0.5
        self.background_mortality = 0.0

        # not given from netlogo interface
        self.q_acceleration = 0
        self.dq_acceleration = 0
        self.h_rate = 0
        self.dh_rate = 0
        self.age_day = 0

        # I added these because they are required for conversion
        self.p_m = None
        self.zoom = None
        self.kap_int = None
        self.E_H__b = None
        self.E_H__p = None
        self.E_G = None
        self.v_rate_int = None

        # self.tau_in_sec = tau_in_sec
        # self.sim_steps_per_tau = tau_in_sec / sim_dt
        # self.dt_in_days = sim_dt / (60 * 60 * 24)
        # # self.parse_day_in_steps = 24 * 60
        # # self.dt_relative_to_simulation = 10 * 60 * 60 * 24 / self.parse_day_in_steps
        # # self.dt = 1 / self.parse_day_in_steps
        # self.d = 1  # in mg/mm**3, same for structure and reserve
        # # Parameters
        # self.Gamma = 21.21  # Gamma=[E_V]= [E_G]*k_G = 4.4*0.8=3.52 in J/mm^3 # Weird! it is 21.21  in Lllandres
        # self.Qm = 4 * self.dt_in_days  # Qm=p_am /Gamma=2.27/3.52=0.645 mm/day
        # self.v = 0.3 * self.dt_in_days  # 0.1817 mm/day for Aedes, 0.2 generalised
        # self.y = 0.99  # This is the yield VE, the ratio of masses of structure and reserve during growth. I am not yet sure but it seems to be equivalent to growth_coefficient :0.8083 for Aedes, 0.8 generalised.
        # self.k = 0.673  # 0.673 for Aedes, 0.8 generalised
        # self.Kw = 0.03 * self.dt_in_days  # 0.1079 J/d.mm^3 for Aedes divided by Gamma = 3.52=> Kw=0.03, 0.018 generalised => Kw=0.005.  vol-spec_som maint = Kw*Gamma
        # self.Kh = 0.002 * self.dt_in_days  # as for Aedes
        #
        # self.H_thr = 0.01
        # self.R_thr = 1.3
        # # W_0 = 0.03  # This is in J as in Llandres
        # L_0 = np.cbrt(W_0 / self.Gamma)
        #
        # self.U_th = 10
        # self.hunger_max = 1
        #

        if self.species == 'aedes':
            self.aedes()
        elif self.species == 'drosophila':
            self.drosophila()
        elif self.species == 'default':
            with open(default_deb_path) as tfp:
                species = json.load(tfp)
            self.__dict__.update(species)
        else:
            self.__dict__.update(self.species)

        self.convert_parameters()
        self.f = 0
        self.lay_egg = False

        self.X = self.J_XAm_rate_int / self.F_m  # set initial value of prey to their carrying capacity

        # individual-variability  ; first their individual variability in the parameter is set
        self.individual_variability()
        #   calc-embryo-reserve-investment     ; then the initial energy is calculated for each
        self.calc_embryo_reserve_investment()
        self.d_E_ref = self.compute_reserve_density()
        self.d_E = self.d_E_ref
        self.hunger = self.compute_hunger()
        self.W = self.compute_wet_weight()

    def compute_reserve_density(self):
        # return self.U_E / self.U_V
        return self.U_E / (self.L ** 3)

    def compute_hunger(self):
        h = np.clip(1 - 0.5 * self.d_E / self.d_E_ref, a_min=0, a_max=1)
        return h

    def run(self, f=None):
        self.tick_counter += 1
        if f is None:
            f = 1
        self.f = f
        # first all individuals calculate the change in their state variables based on the current conditions
        self.calc_dU_E(f=self.f)
        self.calc_dU_H()
        self.calc_dU_R()
        self.calc_dL()

        # if the ageing submodel is turned on, the change in damage inducing compound and damage are calculated
        if self.aging:
            self.calc_dq_acceleration()
            self.calc_dh_rate()

        # if food-dynamics = "logistic"     ; if prey dynamics are set to "logistic" the change in prey density is calculated
        #   [ask patches [calc-d_X]]
        #
        # the the state variables of the individuals and prey are updated based on the delta value
        self.update()

        # ask turtles with [U_H >= U_H^p] ;
        # mature individual check if they have enough energy in their reproduction buffer to repdroduce
        if self.U_H >= self.U_H__e:
            self.calc_lay_eggs()
            if self.lay_egg:
                # if so, they calculate how much energy to invest in an embryo
                self.calc_embryo_reserve_investment()
                # and they produce one offspring
                self.lay_eggs()

        #     do-plots                          ; then the plots are updated
        #   if count turtles = 0 [stop]
        self.age_day += 1 / self.steps_per_day

    # change in reserves: determined by the difference between assimilation (S_A) and mobilization (S_C) fluxes
    # ; when food-dynamics are constant f = the value of f_scaled set in the user interface
    # ; if food is set to  "logistic" f depends on prey density and the half-saturation coefficient (K)
    # ; for embryos f = 0 because they do not feed exogenously
    #
    # to calc-dU_E
    #
    #   if food-dynamics = "constant"
    #   [ ifelse U_H <= U_H^b
    #     [set f 0]
    #     [set f f_scaled]
    #   ]
    #   if food-dynamics = "logistic"
    #   [ ifelse U_H <= U_H^b
    #     [set f 0]
    #     [set f X / (K + X)]
    #   ]

    def calc_dU_E(self, f):
        if not self.U_H <= self.U_H__b:
            self.f = f
        else:
            self.f = 0
        self.e_scaled = self.v_rate * (self.U_E / self.L ** 3)
        self.S_C = self.L ** 2 * (self.g * self.e_scaled / (self.g + self.e_scaled)) * (
                1 + (self.L / (self.g * (self.v_rate / (self.g * self.k_M_rate)))))
        self.S_A = self.f * self.L ** 2
        self.dU_E = self.S_A - self.S_C

        # change in maturity is calculated (for immature individuals only)

    def calc_dU_H(self):
        if self.U_H < self.U_H__b:  # they only invest into maturity until they reach puberty
            self.dU_H = (1 - self.kap) * self.S_C - self.k_J_rate * self.U_H
        elif self.U_H__b <= self.U_H < self.U_H__e:
            if self.embryo and not self.larva:
                self.embryo = False
                self.larva = True
                if self.print_stage_change:
                    print(f'Larval stage reached after {self.age_day} days')
            if self.puppa:
                self.dU_H = (1 - self.kap) * self.S_C - self.k_J_rate * self.U_H
            else:
                self.dU_H = 0

        elif self.U_H__e <= self.U_H:
            if self.puppa and not self.imago:
                self.puppa = False
                self.imago = True
                if self.print_stage_change:
                    print(f'Imago stage reached after {self.age_day} days')
            self.dU_H = 0

    # the following procedure calculates change in reprobuffer if mature
    def calc_dU_R(self):
        if self.larva and self.U_R < self.U_R__p:
            self.dU_R = (1 - self.kap) * self.S_C - self.k_J_rate * self.U_R__p
        elif self.U_R >= self.U_R__p:
            if self.larva and not self.puppa:
                self.larva = False
                self.puppa = True
                if self.print_stage_change:
                    print(f'Puppal stage reached after {self.age_day} days')
            if self.imago:
                self.dU_R = (1 - self.kap) * self.S_C - self.k_J_rate * self.U_R__p
            else:
                self.dU_R = 0

    # the following procedure calculates change in structural length, if growth in negative the individual does not have enough energy to pay somatic maintenance and the starvation submodel is run
    # where growth is set to 0 and individuals divirt enough energy from development (for juveniles) or reprodution (for adults) to pay maintenance costs
    def calc_dL(self):
        self.dL = (1 / 3) * (((self.v_rate / (self.g * self.L ** 2)) * self.S_C) - self.k_M_rate * self.L)
        # if growth is negative use starvation strategy 3 from the DEB book
        if self.e_scaled < self.L / (self.v_rate / (self.g * self.k_M_rate)):
            self.dl = 0
            if self.U_H < self.U_H__p:
                self.dU_H = (
                                    1 - self.kap) * self.e_scaled * self.L ** 2 - self.k_J_rate * self.U_H__p - self.kap * self.L ** 2 * (
                                    self.L / (self.v_rate / (self.g * self.k_M_rate)) - self.e_scaled)
            else:
                self.dU_R = (
                                    1 - self.kap) * self.e_scaled * self.L ** 2 - self.k_J_rate * self.U_H__p - self.kap * self.L ** 2 * (
                                    self.L / (self.v_rate / (self.g * self.k_M_rate)) - self.e_scaled)
            self.dU_E = self.S_A - self.e_scaled * self.L ** 2
            if self.U_H < self.U_H__p:
                if self.dU_H < 0:
                    self.die()
                if self.U_R < 0:
                    self.die()

    # the following procedure calculates the change in damage enducing compounds of an individual
    def calc_dq_acceleration(self):
        self.dq_acceleration = (self.q_acceleration * (self.L ** 3 / (
                self.v_rate / (self.g * self.k_M_rate)) ** 3) * self.sG + self.h_a) * self.e_scaled * (
                                       (self.v_rate / self.L) - ((3 / self.L) * self.dL)) - (
                                       (3 / self.L) * self.dL) * self.q_acceleration

    # the following procedure calculates the change in damage in the individual
    def calc_dh_rate(self):
        self.dh_rate = self.q_acceleration - ((3 / self.L) * self.dL) * self.h_rate

    def convert_parameters(self):
        self.p_am = self.p_m * self.zoom / self.kap_int
        self.U_H__b_int = self.E_H__b / self.p_am
        self.U_H__e_int = self.E_H__e / self.p_am
        self.U_R__p_int = self.E_R__p / self.p_am
        self.k_M_rate_int = self.p_m / self.E_G
        self.g_int = (self.E_G * self.v_rate_int / self.p_am) / self.kap_int

    def individual_variability(self):
        # ; individuals vary in their DEB paramters on a normal distribution with a mean on the input paramater and a coefficent of variation equal to the cv
        #   ; set cv to 0 for no variation
        #   set scatter-multiplier e ^ (random-normal 0 cv)
        scatter_multiplier = np.exp(np.random.normal(0, self.cv))
        self.J_XAm_rate = self.J_XAm_rate_int * scatter_multiplier
        self.g = self.g_int / scatter_multiplier
        self.U_H__b = self.U_H__b_int / scatter_multiplier
        self.U_R__p = self.U_R__p_int / scatter_multiplier
        self.U_H__e = self.U_H__e_int / scatter_multiplier

        self.v_rate = self.v_rate_int
        self.kap = self.kap_int
        self.kap_R = self.kap_R_int
        self.k_M_rate = self.k_M_rate_int
        self.k_J_rate = self.k_J_rate_int
        self.K = self.J_XAm_rate / self.F_m

    def calc_embryo_reserve_investment(self):

        self.L = self.L_0
        self.U_E = self.U_E__0
        self.U_H = 0
        self.U_R = 0
        self.dU_R = 0
        self.U_V = self.compute_structure()

    def calc_lay_eggs(self):
        pass

    def lay_eggs(self):
        pass

    def die(self):
        self.alive = False
        if self.print_stage_change:
            print(f'Dead after {self.age_day} days')

    # the following procedure calculates change in prey density this procedure is only run when prey dynamics are set to "logistic" in the user interface
    #    set d_X (r_X) * X * (1 - (X / K_X))   - sum [ S_A * J_XAm_rate   ] of turtles-here / volume
    def calc_d_X(self):
        pass

    # to update
    # ; individuals update their state variables based on the calc_state variable proccesses
    #   ask turtles
    #   [
    #     set U_E U_E + dU_E / timestep
    #     set U_H U_H + dU_H / timestep
    #     set U_R U_R + dU_R    / timestep
    #     set L L + dL    / timestep
    #     if U_H > U_H^b
    #     [ set q_acceleration q_acceleration + dq_acceleration  / timestep
    #       set h_rate h_rate + dh_rate  / timestep
    #     ]
    #
    #    if aging = "on" [if ticks mod timestep = age-day [if random-float 1 < h_rate [die]] ] ;ageing related mortality
    #    if aging = "off" [if ticks mod timestep = age-day [if random-float 1 < background-mortality [die]] ]
    #  ]
    #   if food-dynamics = "logistic"[ ask patches [ set X X + d_X / timestep]]
    def update(self):
        self.U_E = self.U_E + self.dU_E / self.steps_per_day
        self.U_H = self.U_H + self.dU_H / self.steps_per_day
        self.U_R = self.U_R + self.dU_R / self.steps_per_day
        self.L = self.L + self.dL / self.steps_per_day
        self.U_V = self.compute_structure()
        self.d_E = self.compute_reserve_density()
        self.hunger = self.compute_hunger()
        self.W = self.compute_wet_weight()
        if self.U_H >= self.U_H__b:
            self.q_acceleration = self.q_acceleration + self.dq_acceleration / self.steps_per_day
            self.h_rate = self.h_rate + self.dh_rate / self.steps_per_day
        # ageing related mortality
        if self.aging:
            if self.tick_counter % self.steps_per_day == 0:
                if np.random.uniform(0, 1) < self.h_rate:
                    self.die()
        # background mortality
        else:
            if self.tick_counter % self.steps_per_day == 0:
                if np.random.uniform(0, 1) < self.background_mortality:
                    self.die()
        #   if food-dynamics = "logistic"[ ask patches [ set X X + d_X / timestep]]

    def get_f(self):
        return self.f

    def aedes(self):
        # First three parameters are estimations. Other para,eters from Add_my_pet
        self.U_E__0 = 0.001
        # set initial length to some very small value (embryos start off as nearly all reserves)
        self.L_0 = .00001
        self.d = 1  # in g/cm**3, same for structure and reserve

        self.p_m = 23.57
        self.E_G = 4400
        self.E_H__b = 0.00328786
        self.E_R__p = 0.0166888
        self.E_H__e = 0.05218
        self.zoom = 9.63228

        self.v_rate_int = 0.00581869
        self.kap_int = 0.631228
        self.kap_R_int = 0.95
        self.k_J_rate_int = 0.002
        # self.k_M_rate_int = 0.005356818181818182
        # self.g_int = 0.11276886658823254
        # self.U_H__b_int = 3.580685919765063E-6
        # self.U_H__p_int = 4.64005025218387E-5

        self.shape_factor = 0.352745

        self.h_a = 0.004218
        self.sG = 0.0001

    def drosophila(self):  # This configuration was the best fit in deb_fit.py
        self.U_E__0 = 0.009
        self.L_0 = 0.0054
        self.d = 0.517  # in g/cm**3, same for structure and reserve

        self.p_m = 92
        self.E_G = 9930
        self.E_H__b = 0.0019
        self.E_R__p = 0.347
        self.E_H__e = 0.11
        self.zoom = 8.477

        self.v_rate_int = 0.0125
        self.kap_int = 0.644
        self.kap_R_int = 0.91
        self.k_J_rate_int = 0.0023

        self.shape_factor = 7.6565

        self.h_a = 0.0039
        self.sG = 0.00014

    def compute_wet_weight(self):
        # reserve_volume = self.U_E / self.E_G
        # total_volume = reserve_volume + (self.L * self.shape_factor) ** 3
        # wet_weight_in_mg = total_volume * self.d
        # return wet_weight_in_mg
        physical_V = (self.L * self.shape_factor) ** 3  # in cm**3
        w = physical_V * self.d  # in g
        return w

    # def compute_real_length(self, structural_length):
    #     l = structural_length ** (3 / 2) * 1
    #     return l

    def get_h_rate(self):
        return self.h_rate

    def get_q_acceleration(self):
        return self.q_acceleration

    def get_real_L(self):
        # Structural L is in cm. We turn it to m
        l = self.L * self.shape_factor * 10 / 1000
        return l

    def get_L(self):
        return self.L

    def get_W(self):
        return self.W

    def get_U_E(self):
        return self.U_E

    def get_U_R(self):
        return self.U_R

    def get_U_H(self):
        return self.U_H

    def get_U_V(self):
        return self.U_V

    def get_reserve_density(self):
        return self.d_E

    def reach_stage(self, stage='larva'):
        if stage == 'larva':
            while self.alive and not self.larva:
                f = 1
                self.run(f)

    def compute_structure(self):
        return self.L ** 3 * self.E_G


def deb_default(starvation_days=[]):
    steps_per_day = 24 * 60
    deb = DEB(species='default', steps_per_day=steps_per_day, cv=0, aging=True)
    ww = []
    U_E = []
    U_d = []
    h = []
    L = []
    real_L = []
    U_H = []
    U_R = []
    U_V = []
    fs = []

    c0 = False
    while not deb.puppa:
        if not deb.alive:
            raise ValueError('The default deb larva died.')
        if any([r1 < deb.age_day < r2 for [r1, r2] in starvation_days]):
            f = 0
        else:
            f = 1
        ww.append(deb.get_W() * 1000)
        h.append(deb.hunger)
        real_L.append(deb.get_real_L() * 1000)
        L.append(deb.get_L())
        U_E.append(deb.get_U_E() * 1000)
        U_d.append(deb.get_reserve_density() / 1000)
        U_H.append(deb.get_U_H() * 1000)
        U_R.append(deb.get_U_R() * 1000)
        U_V.append(deb.get_U_V() * 1000)
        fs.append(deb.get_f())
        deb.run(f)
        if deb.larva and not c0:
            c0 = True
            t0 = deb.age_day
        t1 = deb.age_day
    if starvation_days is None :
        id = 'ad libitum'
    elif len(starvation_days)==1 :
        range=starvation_days[0]
        dur=np.round(range[1]- range[0],2)
        id = f'starved {dur} days'
    else :
        id = f'starved {len(starvation_days)} intervals'
    dict = {'birth': t0 * 24,
            'puppation': t1 * 24,
            'mass': ww,
            'length': real_L,
            'reserve': U_E,
            'reserve_density': U_d,
            'hunger': h,
            'structural_length': L,
            'maturity': U_H,
            'reproduction': U_R,
            'structure': U_V,
            'Nticks': deb.tick_counter,
            'f': fs,
            'id' : id}
    return dict


