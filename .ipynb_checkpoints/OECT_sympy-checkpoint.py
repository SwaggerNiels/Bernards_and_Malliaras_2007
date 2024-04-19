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

init_session(quiet=True, use_latex=True, )
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
    
    def fp_plot(self, x, f, eval_pars_sliders=[None], xlim=None, ylim=None):
        '''
        Plots a function, f, vs one of its variables, x, and uses the rest as sliders, eval_pars_sliders.
        x should contain the symbol and its limits e.g.: "(V_DS, 0, 1)"
        '''
        
        if eval_pars_sliders != [None]:
            eval_pars = [x[0]] + list(eval_pars_sliders.keys())
        else: 
            eval_pars = [x[0]]
        
        parameterized_f = self.fp(
                f,
                eval_pars,
            )

        widget = make_widget(
            x,
            parameterized_f,
            eval_pars_sliders,
            xlim=xlim,
            ylim=ylim,
        )

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
            'D_p'   : (1e-10*cm**2, 'hole diffusion', 'm^{2}/s'), #gkoupidenis2023
            'mu_p'  : (1e2*cm**2, 'hole mobility', 'm^{2}/(V·s)'), #bernards2007
            'p'     : (1e18/cm**3   , 'hole concentration' , 'm^{-3}'), #own estimate from bernards2007
            'c_d'   : (15e-7/cm**2, 'double layer capacitance', 'F/m^{2}'), #own value to fit V_P on p from range in bernards2007

            'W'     : (6*mm, 'width of channel', 'm'), #bernards2007
            'L'     : (5*mm, 'length of channel', 'm'), #bernards2007
            'd'     : (62.5*nm, 'thickness of channel', 'm'), #bernards2007 calculated from conduction value 1.2e-4 [S] 

            'V_G'   : (0, 'gate voltage', 'V'),
            'V_DS'  : (.5, 'drain-source voltage', 'V'),

            'V_P'   : (1.23, 'pinch voltage', 'V'), #bernards2007 fig3
            # 'G'     : (1.2e-4, 'channel conductivity', 'S'), #bernards2007
            }

        self.parameter_values, self.parameter_names, self.parameter_units = self.init_variables(self.parameters_init)
        self.parameter_default_values = self.parameter_values

        # Categorical parameters
        self.material = 'PEDOT:PSS' # material name

        x = Symbol('x')
        Vx = self.V(x)
        dx = Symbol('dx')
        # Qx = self.Q(dx, Vx)
        self.curves = {
            'Ids' : self.fp_plot(
                        (self.V_DS,-2,2),
                        self.I_SD(),
                        {self.V_G: (0,0,2)},
                        xlim=(-2,2),
                        ylim=(-1e-4,1e-4),
                    ),

            'V(x)' : self.fp_plot(
                        (x,0,self.fp(self.L)),
                        Vx,
                        {self.V_DS: (0,-2,2)},
                        xlim=(0,self.fp(self.L)),
                        ylim=(-2,2),
                    ),

            # 'Q(x)' : self.fp_plot(
            #             (x,0,self.fp(self.L)),
            #             Qx,
            #             {
            #                 dx: (self.fp(self.L)/100,self.fp(self.L)/100,self.fp(self.L)/10),
            #                 self.V_DS: (0,-2,2),
            #                 self.V_G: (0,0,2),
            #             },
            #             xlim=(0,self.fp(self.L)),
            #             ylim=(-1e-7,1e-7),
            #         ),
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
            Vx = self.V(x)

        V_DS    = self.V_DS
        L       = self.L

        x = Symbol('x')
        f = Derivative(Vx,x)

        return f

    def V(self, x=Symbol('x')):
        V_DS    = self.V_DS
        L       = self.L

        f = Piecewise(
            (
                (x/L) * V_DS,
                ((0 < x) & (x < L))
            ))

        return f

    # def Q(self, dx=Symbol('dx'), Vx=None):
    #     c_d     = self.c_d
    #     W       = self.W
    #     V_G     = self.V_G

    #     L       = self.L
    #     discretized_N = L/dx
    #     # f = c_d * W * dx * (V_G - Vx)
    #     f = Piecewise(
    #         *[(
    #             c_d * W * dx * (V_G - Vx),
    #             (((i-.5)*dx < x) & (x < (i+.5)*dx))
    #         ) for i in range(10)])

    #     return f

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
        p = self.p
        mu_p = self.mu_p
        dVdx = self.dVdx(Vx)

        f = q * mu_p * p * dVdx
        
        return f
    
    def J(self, Vx=None):
        '''Total flux'''
        q       = self.q
        mu_p    = self.mu_p
        p       = self.p
        V_G      = self.V_G
        V_DS     = self.V_DS
        V_P      = self.V_P
        dVdx     = self.dVdx(Vx)

        
        f = q * mu_p * p * ( 1 - (V_G-V_DS)/V_P ) * dVdx
        
        return f
    
    def _J(self, Vx=None):
        '''Total flux'''
        if Vx == None:
            x = Symbol('x')
            Vx = self.V(x)

        q       = self.q
        mu_p    = self.mu_p
        p       = self.p
        V_G      = self.V_G
        V_DS     = self.V_DS
        V_P      = self.V_P
        dVdx     = self.dVdx(Vx)
        L        = self.L
        
        f = q * mu_p * p * ( 1 - (V_G-V_DS)/V_P ) * dVdx
        
        return f
    
    def _I(self, Vx=None):
        '''Total flux'''
        if Vx == None:
            x = Symbol('x')
            Vx = self.V(x)

        q       = self.q
        mu_p    = self.mu_p
        p       = self.p
        V_G      = self.V_G
        V_DS     = self.V_DS
        V_P      = self.V_P
        dVdx     = self.dVdx(Vx)
        W        = self.W
        d        = self.d
        L        = self.L
        
        f = W*d*self._J()
        
        return f
    
    def J_Rdrop(self, Vx=None):
        '''Total flux'''
        q       = self.q
        mu_p    = self.mu_p
        p       = self.p
        V_G      = self.V_G
        V_DS     = self.V_DS
        V_P      = self.V_P
        dVdx     = self.dVdx(Vx)

        
        f = q * mu_p * p * ( 1 - (V_G-V_DS)/V_P ) * dVdx
        
        return f

    def _fplot():
        '''TODO'''
        if plot:
            p1 = spplot(
                J, 
                (p,0,1e20),
                xlabel = r'$p (m^{-3})$',
                ylabel = r'$J_e (\mu A)$',
                )

    def G(self):
        q       = self.q
        mu_p    = self.mu_p
        p      = self.p
        W      = self.W
        L      = self.L
        d      = self.d
        
        G = q * mu_p * p * W * d / L
        return(G)       
    
    def V_P_func(self):
        q       = self.q
        p       = self.p
        d       = self.d
        c_d     = self.c_d
        
        V_P = q * p * d / c_d
        return(V_P)       

    def I_SD(self):
        V_DS    = self.V_DS
        V_G     = self.V_G
        V_P     = self.V_P
        G       = self.G()

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
        
        return(I_SD)    

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


# fig3 = oect.figure_3()
# fig4 = oect.figure_4()
# fig5 = oect.figure_5()


# from here call: fig3

something = '''
import ipywidgets as widgets
from IPython.display import display, HTML, IFrame
import networkx as nx

nx_graph = nx.cycle_graph(10)
nx_graph.nodes[1]['title'] = 'Number 1'
nx_graph.nodes[1]['group'] = 1
nx_graph.nodes[3]['title'] = 'I belong to a different group!'
nx_graph.nodes[3]['group'] = 10
nx_graph.add_node(20, size=20, title='couple', group=2)
nx_graph.add_node(21, size=15, title='couple', group=2)
nx_graph.add_edge(20, 21, weight=5)
nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
nt = Network('500px', '500px')
# populates the nodes and edges data structures
nt.from_nx(nx_graph)

net = make_network()
iframe = show_network(nt)
HTML(iframe)'''



