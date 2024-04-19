import importlib
import matplotlib.pyplot as plt
plt.close()
import numpy as np
import seaborn as sns
from sympy import *
from sympy.plotting.plot import plot,Plot

# Make sure to: set PYTHONPATH=.
import lib.widget_display_lib
from lib.widget_display_lib import *
importlib.reload(lib.widget_display_lib)

import lib.widget_network_lib
from lib.widget_network_lib import *
importlib.reload(lib.widget_network_lib)

# init_session(quiet=True, use_latex=True, )
# init_printing(use_latex=True)

class DeviceModel():
    def __init__(self):
        self.constants_init = {
            'q'     : (1.6e-19, 'elementary charge', 'C'),
            'V_t'   : (25.69e-3, 'thermal voltage', 'V'),
        }

        self.constants_values, self.constants_names, self.constants_units = self.init_variables(self.constants_init)

    def set_variable(self,parameter_string):
        setattr(self, parameter_string, Symbol(parameter_string))

    def init_variables(self, inits):
        v = {}
        n = {}
        u = {}
        
        for key in inits.keys():
            self.set_variable(key)
            v[key] = inits[key][0]
            n[key] = inits[key][1]
            u[key] = inits[key][2]

        return(v,n,u)
    
    def fp(self, f, eval_pars=[None], constants_replace=True):
        '''Function for evaluating the equation for only the desired parameters, 
        others are replaced by numerical value.
        Returns the sympy expression with that allows the mapping for the desired input parameters'''
        eval_pars = list(eval_pars)
        
        mapping = []
        for par_name,val in self.parameter_values.items():
            par = getattr(self,par_name)

            if par in eval_pars:
                continue
            else:
                mapping.append((par,val))

        if constants_replace:
            for constant_name,val in self.constants_values.items():
                constant = getattr(self,constant_name)

                if constant in eval_pars:
                    continue
                else:
                    mapping.append((constant,val))


        f_eval = f.subs(mapping)

        return f_eval
    
    def fp_plot(self, x, f, eval_pars_sliders=[None], xlim=None, ylim=None, **kwargs):
        '''
        Plots a function, f, vs one of its variables, x, and uses the rest as sliders, eval_pars_sliders.
        x should contain the symbol and its limits e.g.: "(V_DS, 0, 1)"
        '''
        if xlim==None:
            xlim = x[1:]
        
        if eval_pars_sliders != [None]:
            eval_pars = [x[0]] + list(eval_pars_sliders.keys())
        else: 
            eval_pars = [x[0]]
        
        parameterized_f = self.fp(
                f,
                eval_pars,
            )

        # if no sliders are specified make one for each free variable except the x-axis variable
        if eval_pars_sliders == [None]:
            eval_pars_sliders = {symbol : (1, -10, 10, 20) for symbol in parameterized_f.free_symbols}
            if x[0] in eval_pars_sliders:
                del eval_pars_sliders[x[0]]

        widget = make_widget(
            x,
            parameterized_f,
            eval_pars_sliders,
            xlim=xlim,
            ylim=ylim,
            **kwargs
        )
        widget.equation = parameterized_f

        return widget

    def fp_plot(self, x, f, eval_pars_sliders=[None], xlim=None, ylim=None, **kwargs):
        '''
        Plots a group of functions, f, vs one of its variables, x, and uses the rest as sliders, eval_pars_sliders.
        x should contain the symbol and its limits e.g.: "(V_DS, 0, 1)"
        '''
        if xlim==None:
            xlim = x[1:]
        
        if eval_pars_sliders != [None]:
            eval_pars = [x[0]] + list(eval_pars_sliders.keys())
        else: 
            eval_pars = [x[0]]
        
        parameterized_f = self.fp(
                f,
                eval_pars,
            )

        # if no sliders are specified make one for each free variable except the x-axis variable
        if eval_pars_sliders == [None]:
            eval_pars_sliders = {symbol : (1, -10, 10, 20) for symbol in parameterized_f.free_symbols}
            if x[0] in eval_pars_sliders:
                del eval_pars_sliders[x[0]]

        widget = make_widget(
            x,
            parameterized_f,
            eval_pars_sliders,
            xlim=xlim,
            ylim=ylim,
            **kwargs
        )
        widget.equation = parameterized_f

        return widget

    def get_lamba(self, f, eval_pars=None):
        '''Function for evaluating the equation for only the desired parameters, 
        others are replaced by numerical value.
        Returns the lambda function with that allows the mapping for the desired input parameters'''
        eval_pars = list(eval_pars)
        
        pars = []
        for par_name,val in self.parameter_values.items():
            par = getattr(self,par_name)

            if par in eval_pars:
                pars.append(par)
        
        f_eval = self.fp(f, eval_pars)
        f_lambda = lambdify(pars,f_eval)

        return f_lambda

# Library OECT models
class SimpleOECT(DeviceModel):
    '''
    Contains all the OECT model taken from:
        Bernards and Malliaras, “Steady-State and Transient Behavior of Organic Electrochemical Transistors.”
        It is a simple 1D diffusion model.
    '''
    def __init__(self):
        super().__init__()

        cm = 0.01
        mm = 1e-3
        um = 1e-6
        nm = 1e-9

        # Numerical parameters
        self.parameters_init = {
            'D_p'   : (1e-10*cm**2, 'channel hole diffusion', 'm^{2}/s'), #gkoupidenis2023
            'mu_p'  : (1e-2*cm**2, 'channel hole mobility', 'm^{2}/(V·s)'), #bernards2007
            'p0'     : (1e21/cm**3   , 'hole concentration' , 'm^{-3}'), #own estimate from bernards2007
            'c_d'   : (15e-7/cm**2, 'double layer capacitance', 'F/m^{2}'), #own value to fit V_P on p0 from range in bernards2007
            'R_s'   : (1e8, 'series resistance', 'Ohm'), #own value

            'W'     : (6*mm, 'width of channel', 'm'), #bernards2007
            'L'     : (5*mm, 'length of channel', 'm'), #bernards2007
            'T'     : (62.5*nm, 'thickness of channel', 'm'), #bernards2007 calculated from conduction value 1.2e-4 [S] 
            'dx'     : (0.5*mm, 'discretization length of channel', 'm'), #own value based on L/10

            'V_G'   : (0, 'gate voltage', 'V'),
            'V_DS'  : (.5, 'drain-source voltage', 'V'),

            'V_P'   : (1.23, 'pinch voltage', 'V'), #bernards2007 fig3
            # 'G'     : (1.2e-4, 'channel conductivity', 'S'), #bernards2007
            }

        self.parameter_values, self.parameter_names, self.parameter_units = self.init_variables(self.parameters_init)
        self.parameter_default_values = self.parameter_values

        # Categorical parameters
        self.material = 'PEDOT:PSS' # material name

        self.cvars = {} # Default curve variables
        self.make_cvars()
        self.curves = {} # Default curves to use
        self.make_curves()

    def make_cvars(self):
        x = self.cvars['x'] = Symbol('x')
        Vx = self.cvars['Vx'] = self.V(x)
        Ex = self.cvars['Ex'] = self.dVdx(Vx)
        Qx = self.cvars['Qx'] = self.Q(Vx)
        px = self.cvars['px'] = self.p(Qx)
        Jx = self.cvars['Jx'] = self.J(Vx)
        
        f = self.cvars['f'] = Symbol('f')
        t = self.cvars['t'] = Symbol('t')
        t0 = self.cvars['t0'] = Symbol('t0')
        t1 = self.cvars['t1'] = Symbol('t1')
        dt = self.cvars['dt'] = Symbol('dt')
        Qt = self.cvars['Qt'] = self.Q(t=t)
        pt = self.cvars['pt'] = self.p(Qt=Qt)
        expt = self.cvars['expt'] = self.exp(t=t, t0=t0)
        
        IGmag = self.cvars['I_G_magnitude'] = Symbol('IGmag')
        IGt = self.cvars['IGt'] = self.I_Gt(t=t, t0=t0, t1=t1, I_G_magnitude=IGmag)
        ISDtIG = self.cvars['ISDtIG'] = self.I_SD(t=t, I_G=IGt, f=f, t0=t0, t1=t1)
        VGmag = self.cvars['V_G_magnitude'] = Symbol('VGmag')
        VGt = self.cvars['VGt'] = self.V_Gt(t=t, t0=t0, t1=t1, V_G_magnitude=VGmag)
        ISDtVG = self.cvars['ISDtVG'] = self.I_SD(t=t, V_G=VGt, f=f, t0=t0, t1=t1)

    def make_curves(self):
        xleft = -.5*self.fp(self.L)
        xright = 1.5*self.fp(self.L)
        xlim = (xleft,xright)
        
        tleft = -1
        tright = 20
        tright2 = 60
        tlim = (tleft,tright)
        tlim2 = (tleft,tright2)

        tau_i = self.fp(self.tau_i())

        # self.plot_groups = {
        #     "Channel electrical" : (
        #         self.V,
        #         'E(x)',
        #         'Q(x)',
        #         'J(x)',
        #     ),

        #     "Steady state current" : (
        #         (self.I_SD, 
        #             (self.V_DS,-2,2)),
        #         'fig3(V_DS)',
        #     ),

            
        # }


        self.curves = {

            'I_SD(V_DS)' : self.fp_plot(
                (self.V_DS,-2,2),
                self.I_SD(),
                {
                    self.V_G: (0,0,2)
                },
                ylim=(-1e-4,3e-4),
                ylabel='$I_{SD}$',
                title='$I_{SD}(V_{DS})$',
            ),

            'fig3' : self.fp_plot(
                (self.V_DS, -.5, .5),
                self.I_SD(),
                {
                    self.V_G : (0, 0, 0.6, 3)
                },
                ylim=(-70e-6,70e-6),
                ylabel = '$I_{sd} [\mu A]$',
            ),   

            'V(x)' : self.fp_plot(
                (self.cvars['x'],*xlim),
                self.cvars['Vx'],
                {
                    self.V_DS: (0,-2,2)
                },
                ylim=(-2.5,2.5),
                ylabel='$V(x)$',
            ),

            'E(x)' : self.fp_plot(
                (self.cvars['x'],*xlim),
                self.cvars['Ex'],
                {
                    self.V_DS: (0,-2,2)
                },
                ylim=(-5e3,5e3),
                ylabel='$E(x)$',
            ),

            'Q(x)' : self.fp_plot(
                (self.cvars['x'],*xlim),
                self.cvars['Qx'],
                {
                    # self.cvars['dx']: (self.fp(self.L)/10,self.fp(self.L)/100,self.fp(self.L)/10),
                    self.V_DS: (0,-2,2),
                    self.V_G: (0,0,2),
                },
                ylim=(-2e-7,2e-7),
                ylabel='$Q(x)$',
            ),
                        
            'p(x)' : self.fp_plot(
                (self.cvars['x'],*xlim),
                self.cvars['px'],
                {
                    self.V_DS: (0,-2,2),
                    self.V_G: (0,0,2),
                },
                ylim=(1e-30,1e30),
                ylabel='$p(x)$',
                yscale='log',
            ),

            'J(x)' : self.fp_plot(
                (self.cvars['x'],*xlim),
                self.cvars['Jx'],
                {
                    self.V_DS: (0,-2,2),
                    self.V_G: (0,0,2),
                },
                ylim=(-0.2e6,2e6),
                ylabel='$J(x)$',
                # yscale='log'
            ),
            
            'I_G(t)' : self.fp_plot(
                (self.cvars['t'],*tlim),
                self.cvars['IGt'],
                {
                    self.cvars['t0']: (0,*tlim, len(range(*tlim))),
                    self.cvars['t1']: (0,*tlim, len(range(*tlim))),
                    self.cvars['I_G_magnitude']: (1e-6,1e-6, 1e-5, 10),
                },
                ylim=(-1.5e-5,1.5e-5),
                ylabel='$I_G(t)$',
            ),

            'V_G(t)' : self.fp_plot(
                (self.cvars['t'],*tlim),
                self.cvars['VGt'],
                {
                    self.cvars['t0']: (0,*tlim, len(range(*tlim))),
                    self.cvars['t1']: (0,*tlim, len(range(*tlim))),
                    self.cvars['V_G_magnitude']: (1,0, 1, 10),
                },
                ylim=(-1.5,1.5),
                ylabel='$V_G(t)$',
            ),

            'I_SD(t,I_G)' : self.fp_plot(
                (self.cvars['t'],*tlim),
                self.cvars['ISDtIG'],
                {
                    self.cvars['t0']: (0,*tlim, len(range(*tlim))),
                    self.cvars['t1']: (0,*tlim, len(range(*tlim))),
                    self.cvars['I_G_magnitude']: (1e-6,1e-6, 1e-5, 10),
                    self.cvars['f']: (0, 0, .5),
                    self.V_DS: (0, -0.1, 0.1)
                },
                # ylim=(-5e-5,5e-5),
                ylabel='$I_{SD}(t,I_G)$',
            ),

            'I_SD(t,V_G)' : self.fp_plot(
                (self.cvars['t'],*tlim2),
                self.cvars['ISDtVG'],
                {
                    self.cvars['t0']: (0,*tlim2, len(range(*tlim))),
                    self.cvars['t1']: (0,*tlim2, len(range(*tlim))),
                    self.cvars['V_G_magnitude']: (1,0, 1, 10),
                    self.cvars['f']: (.5, 0, .5),
                    self.V_DS: (2, -2, 2, 100)
                },
                # ylim=(-10,10),
                xlabel='$t/\\tau_i$',
                ylabel='$I_{SD}(t,V_G)$',
            ),

            'fig7' : self.fp_plot(
                (self.cvars['t'],-1,3),
                self.cvars['ISDtVG'].subs(self.cvars['t0'],0) / ( self.G() * ( 1 - ( (-.5*self.V_DS)/self.V_P) ) * self.V_DS ),
                {
                    self.cvars['t0']: (0,*tlim, len(range(*tlim))),
                    self.cvars['t1']: (0,*tlim, len(range(*tlim))),
                    self.cvars['V_G_magnitude']: (1,0, 1, 10),
                    self.cvars['f']: (.5, 0, .5),
                    self.V_DS: (2, -2, 2)
                },
                # ylim=(-5e-5,5e-5),
                xlabel='$t/\\tau_i$',
                ylabel='$I_{SD}(t,I_G)$',
            ),

            'Q(t)' : self.fp_plot(
                (self.cvars['t'],*tlim),
                self.cvars['Qt'],
                {
                    self.V_DS: (0,-2,2),
                    self.V_G: (0,0,2),
                    self.R_s: (1e6,1,9,51, 'R_{solution}', 'log'),
                },
                ylim=(-2e-7,2e-7),
                ylabel='$Q(t)$',
            ),

            'p(t)' : self.fp_plot(
                (self.cvars['t'],*tlim),
                self.cvars['pt'],
                {
                    self.V_DS: (0,-2,2),
                    self.V_G: (1,0,2),
                    # self.dx: (self.fp(self.L)/10,self.fp(self.L)/1000,self.fp(self.L)/10),
                    self.R_s: (1e6,1,9,51, 'R_{solution}', 'log'),
                },
                # ylim=(1e27,1e28),
                yscale='symlog',
                ylabel='$p(t)$',
            ),

            'exp(t)' : self.fp_plot(
                (self.cvars['t'],*tlim),
                self.cvars['expt'],
                {
                    self.cvars['t0']: (0,*tlim, len(range(*tlim))),
                },
                ylim=(0,1),
                ylabel='$exp(t)$',
            )
        }

    def plot_figures(self):
        '''
        This will give you the figures from their paper.
        '''
        
        # Figure 3
        lines = []
        V_g_values = [0,0.2,0.4,0.6]

        lines=None
        for V_G in V_g_values:
            self.parameter_values['V_G'] = V_G 
            line = plot(
                self.fp(
                    self.I_SD(),
                    [self.V_DS]
                    ),
                (oect.V_DS, -.5, .5),
                show=False,
                backend='matplotlib')
            
            if lines==None:
                lines = line
            else:
                lines.append(line[0])
        
        # plt.legend([f'$V_g = {V_g}$' for V_g in V_g_values])
        
        # plt.xlabel(r'$V_d (V)$')
        # plt.ylabel(r'$I_{sd} (\mu A)$')
        # plt.grid(True)

        lines.show()

        self.parameter_values = self.parameter_default_values

    def diffusion_1D_flux(self, x):
        mu_p    = self.hole_mobility
        p0      = self.hole_density
        V_g      = self.V_G
        V_p      = self.V_p
        
        # J = q*mu_p*p0 * ( 1 - (V_g-V(x))/V_p ) * dV(x)/dx
        # return J
        pass

    def dVdx(self, Vx=None):
        if Vx == None:
            Vx = self.V(self.cvars['x'])

        V_DS    = self.V_DS
        L       = self.L

        # f = Vx*self.cvars['x']**2 
        f = diff(Vx,self.cvars['x'])

        return f

    def V(self, x=Symbol('x')):
        V_DS    = self.V_DS
        L       = self.L

        f = Piecewise(
                (
                    V_DS,
                    ((L < x))
                ),
                (
                    (x/L) * V_DS,
                    ((0 < x) & (x < L))
                ),
                (
                    0,
                    ((x < 0))
                ),
            )

        return f

    def exp(self, t=None, t0=None, percentage=.99):

        t1 = t0+N(-ln(1-percentage))
        f = (Heaviside(t-t0)-Heaviside(t-t1)) * exp(-(t-t0))

        return f

    def Q(self, Vx=None, t=None):
        if t==None:
            c_d     = self.c_d
            W       = self.W
            dx       = self.dx
            V_G     = self.V_G

            f = c_d * W * dx * (V_G - Vx)
            # L       = self.L
            # discretized_N = L/dx
            # f = Piecewise(
            #     *[(
            #         c_d * W * dx * (V_G - Vx),
            #         (((i-.5)*dx < x) & (x < (i+.5)*dx))
            #     ) for i in range(10)])
        
        if t!=None:
            c_d     = self.c_d
            W       = self.W
            dx      = self.dx
            L       = self.L
            R_s     = self.R_s
            V_G     = self.V_G
            V_DS     = self.V_DS

            A = W*L
            C_d = c_d*A
            deltaV = V_G - 0.5*V_DS

            Q_ss = C_d * deltaV
            tau_i = R_s*C_d

            f = Piecewise(
                (
                    Q_ss * ( 1 - exp(-t/tau_i) ),
                    t > 0,
                ),
                (
                    0,
                    t <= 0,
                )
            )

        return f

    def p(self, Qx=None, Qt=None):
        
        p0      = self.p0
        q       = self.q
        L       = self.L
        W       = self.W
        dx      = self.dx
        T       = self.T

        if Qt == None:
            f = p0 * (1 - Qx/(q*p0*(dx*W*T)))
        
        if Qt != None:
            f = Piecewise(
                (
                    p0 * (1 - Qt/(q*p0*(L*W*T))),
                    Qt > 0,
                ),
                (
                    p0,
                    True,
                )
            )

        return f

    def V_Rdrop(self, x, R_fraction):
        V_DS    = self.V_DS
        L       = self.L

        V1 = V_DS*(R_fraction/2)
        V2 = V_DS*(1-R_fraction/2)

        f = Piecewise(
            (
                V1 + (x/L) * (V2-V1),
                ((.25*L < x) & (x < .75*L))
            ))

        return f

    def J_e(self, Vx=None):
        '''Electron flux'''
        q = self.q
        p0 = self.p0
        mu_p = self.mu_p
        dVdx = self.dVdx(Vx)

        f = q * mu_p * p0 * dVdx
        
        return f
    
    def J(self, Vx=None):
        '''Total flux'''
        q       = self.q
        mu_p    = self.mu_p
        p0       = self.p0
        V_G      = self.V_G
        V_DS     = self.V_DS
        V_P      = self.V_P
        dVdx     = self.dVdx(Vx)

        f = q * mu_p * p0 * ( 1 - (V_G-V_DS)/V_P ) * dVdx
        
        return f
    
    def _J(self, Vx=None):
        '''Total flux'''
        if Vx == None:
            x = Symbol('x')
            Vx = self.V(x)

        q       = self.q
        mu_p    = self.mu_p
        p0       = self.p0
        V_G      = self.V_G
        V_DS     = self.V_DS
        V_P      = self.V_P
        dVdx     = self.dVdx(Vx)
        L        = self.L
        
        f = q * mu_p * p0 * ( 1 - (V_G-V_DS)/V_P ) * dVdx
        
        return f
    
    def _I(self, Vx=None):
        '''Total flux'''
        if Vx == None:
            x = Symbol('x')
            Vx = self.V(x)

        q       = self.q
        mu_p    = self.mu_p
        p0       = self.p0
        V_G      = self.V_G
        V_DS     = self.V_DS
        V_P      = self.V_P
        dVdx     = self.dVdx(Vx)
        W        = self.W
        T        = self.T
        L        = self.L
        
        f = W*T*self._J()
        
        return f
    
    def J_Rdrop(self, Vx=None):
        '''Total flux'''
        q       = self.q
        mu_p    = self.mu_p
        p0       = self.p0
        V_G      = self.V_G
        V_DS     = self.V_DS
        V_P      = self.V_P
        dVdx     = self.dVdx(Vx)

        
        f = q * mu_p * p0 * ( 1 - (V_G-V_DS)/V_P ) * dVdx
        
        return f

    def G(self):
        q       = self.q
        mu_p    = self.mu_p
        p0      = self.p0
        W      = self.W
        L      = self.L
        T      = self.T
        
        G = q * mu_p * p0 * W * T / L
        return(G)       
    
    def V_P_func(self):
        q       = self.q
        p0       = self.p0
        T       = self.T
        c_d     = self.c_d
        
        V_P = q * p0 * T / c_d
        return(V_P)       

    def I_Gt(self, t=None, t0=None, t1=None, I_G_magnitude=None):
        if t != None and t0 != None:
            f = I_G_magnitude * (Heaviside(t-t0) - Heaviside(t-t1))

        return(f)
    
    def V_Gt(self, t=None, t0=None, t1=None, V_G_magnitude=None):
        if t != None and t0 != None:
            f = V_G_magnitude * (Heaviside(t-t0) - Heaviside(t-t1))

        return(f)

    def I_SD(self, t=None, V_G=None, I_G=None, f=None, t0=None, t1=None):
        V_DS    = self.V_DS
        V_P     = self.V_P
        if V_G == None:
            V_G     = self.V_G
        if t0 == None:
            t0 = 0
        if t1 == None:
            t1 = 0
        G       = self.G()

        c_d     = self.c_d
        R_s     = self.R_s
        L       = self.L
        W       = self.W
        mu      = self.mu_p

        # Steady state
        if t==None and I_G==None:
            V_DS_sat = V_G-V_P
            
            I_SD = Piecewise(
                ( # completely dedoped (saturation) 3rd quadrant
                    - (G * V_DS_sat**2) / (2*V_P),
                    V_DS<=V_DS_sat
                ),
                ( # pinch-off region (V_DS > V_G), here the channel requires additional biasing to overcome V_DS in order to gate.
                    G * ( V_DS - ((V_G**2)/(2*V_P)) ),
                    V_DS>=V_G
                ),
                ( # parabolic (triode region)
                    G * ( 1 - ((V_G - (.5*V_DS))/V_P) ) * V_DS,
                    V_DS<V_G
                ),
            )

        # Input gate current
        if t!=None and I_G!=None and f!=None:
            self.parameter_values['L'] = 0.5e-3

            tau_e = L**2 / mu * V_DS

            I0 = self.fp(self.I_SD())

            I_SD = I0 - I_G * (f + (t-t0)/tau_e)
        
        # Input gate voltage
        if t!=None and f!=None and I_G==None: # check for I_G here because V_G will always have a value
            # valid only in non-saturation region
            # I_ss     = G * ( 1 - ((V_G - (.5*V_DS))/V_P) ) * V_DS
            I_ss_0   = G * ( 1 - (       (-.5*V_DS)/V_P) ) * V_DS
            I_ss_V_G = G * ( 1 - ((V_G - (.5*V_DS))/V_P) ) * V_DS

            Delta_I_ss = I_ss_0 - I_ss_V_G # G*V_G*V_DS / V_P

            A = W*L
            C_d = c_d*A
            tau_i = R_s*C_d
            tau_e = L**2 / (mu * V_DS)
            print(self.fp(tau_i))
            print(self.fp(tau_e))

            #actual equation is with t -> (t/tau_i) but here this is removed for easier plotting
            I_SD = I_ss_V_G  +  Delta_I_ss * (1 - f*(tau_e/tau_i)) * exp(-(t-t0))
        
        return(I_SD)    

    def tau_i(self):
        c_d     = self.c_d
        R_s     = self.R_s
        L       = self.L
        W       = self.W

        A = W*L
        C_d = c_d*A
        tau_i = R_s*C_d

        return(tau_i)

    def plot_E_drop(self):
        self.parameter_values = self.parameter_default_values

        x = Symbol('x')
        R_fraction = Symbol('R_fraction')
        L = oect.fp(oect.L)

        plot = oect.fp_plot(
            x = (x, .25*L, .75*L),

            f = oect.V_Rdrop(x, R_fraction),
            
            eval_pars_sliders=
            {
                oect.V_DS : (0.5, -1, 1, 10),
                R_fraction : (0, 0, 1, 10),

            },
            xlim=(0,L),
            ylim=(-2,2),
        )

        return(plot)
    
    def figure_3(self):
        self.parameter_values = self.parameter_default_values

        plot = self.fp_plot(
            x = (self.V_DS, -.5, .5),

            f = self.I_SD(),
            
            eval_pars_sliders=
            {
                self.V_G : (0, 0, 0.6, 3)
            },

            xlim=(-.5,.5),
            ylim=(-70e-6,70e-6),
        )

        plt.ylabel('$I_{sd} [\mu A]$')

        return plot
    
    def figure_4(self):
        self.parameter_values = self.parameter_default_values
        L_value = self.parameter_default_values['L']
        self.V_P = 1.1

        G = self.fp(self.G())
        print(f'G = {G:1.2e} [S]')

        R_series = Symbol('R_series')
        VoverR = self.V_DS / R_series
        

        plot = self.fp_plot(
            x = (self.V_DS, -1, 1),

            f = (self.I_SD())/G,

            eval_pars_sliders=
            {
                self.V_G : (0, 0, 0.6, 2),
                self.L : (L_value, L_value/2 , L_value, 1),
                R_series : (0, -1 , 8, 90, 'R_{series}', 'log'),
            },

            xlim=(-1,1),
            ylim=(-.7,1.1),
        )

        return plot
    
    def figure_5(self):
        self.parameter_values = self.parameter_default_values
        self.V_P = 1.1
        G = self.fp(self.G())
        print(f'G = {G:1.2e} [S]')

        G = self.G()
        R_series = Symbol('R_series')
        VoverR = self.V_DS / R_series

        f = self.fp(
            (self.I_SD())/G + VoverR,
            [
                self.V_DS,
                self.V_G,
                self.L,
            ]
        )

        L_value = self.parameter_default_values['L']

        return make_widget(
            (self.V_DS, -1, 1),
            f,
            {
                self.V_G : (0, 0, 0.6, 2),
                self.L : (L_value, L_value/2 , L_value, 1),
                R_series : (0.001, 0.001 , 100, 51, 'R_{series}'),
            },
            xlim=(-1,1),
            ylim=(-.7,1.1),
        )

oect = SimpleOECT()
ISDtVG = oect.curves['I_SD(t,V_G)']
# fig3 = oect.figure_3()
# fig4 = oect.figure_4()
# fig5 = oect.figure_5()
# from here call: fig3

