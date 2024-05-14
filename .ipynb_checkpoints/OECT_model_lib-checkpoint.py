import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

q = 1.6e-19 # elementary charge

# Library OECT models
class OECT_model():
    '''
    Contains all the OECT model functions and can be changed
    '''
    def __init__(self):
        # Default parameters
        self.material = 'PEDOT:PSS' # material name
        self.hole_mobility = 5.55e-5 # hole mobility
        self.hole_density = 1e-10 # hole density
        self.W = 100e-6 # width
        self.T = 10e-6 # thickness
        self.L = 200e-6 # length
        self.dx = 1e-6 # length discretization
        
        self.V_G = 0 # voltage at gate
        self.V_D = 1 # voltage at drain
        self.V_p = 1.23 # pinch-off voltage

        self.G = 1.2e-4 # device conductivity
        
    def diffusion_1D(self, plot=True):
        '''
        This model is taken from:
        Bernards and Malliaras, “Steady-State and Transient Behavior of Organic Electrochemical Transistors.”
        It is a simple 1D diffusion model.
        
        This will give you the figures from their paper.
        '''
        
        # replicate figure 3
        self.material = 'PEDOT:PSS' # material name
        self.hole_mobility = 5.55e-5 # hole mobility
        self.hole_density = 1e-10 # hole density
        self.W = 6e-3 # width
        self.T = 50e-9 # thickness
        self.L = 5e-3 # length
        
        self.V_G = 0 # voltage at gate
        self.V_D = 1 # voltage at drain
        self.V_p = 1.23 # pinch-off voltage
        
        # self.G = self.diffusion_1D_conductivity() # unknown parameters
        self.G = 1.2e-4 # device conductivity [S]
        
        #Show Ids-Vds characteristics
        V_g_values = [0, 0.2, 0.4, 0.6]
        for V_g in V_g_values:
            self.V_G = V_g
            V_d = np.linspace(-.5,.5)
            I = self.diffusion_1D_current(V_d)
            plt.plot(V_d,I * 1e6)
        
        plt.legend([f'$V_g = {V_g}$' for V_g in V_g_values])
        
        plt.xlabel(r'$V_d (V)$')
        plt.ylabel(r'$I_{sd} (\mu A)$')
        plt.grid(True)
        
        plt.show()

    def diffusion_1D_flux(self, x):
        q       = self.q
        mu_p    = self.hole_mobility
        p0      = self.hole_density
        V_g      = self.V_G
        V_p      = self.V_p
        
        # J = q*mu_p*p0 * ( 1 - (V_g-V(x))/V_p ) * dV(x)/dx
        # return J
        pass

    def diffusion_1D_flux_electronic(self, x, plot=False):
        q       = self.q
        mu_p    = self.hole_mobility
        p0      = self.hole_density
        V_g      = self.V_G
        V_p      = self.V_p
        
        J = q*mu_p*p0 * ( 1 - (V_g-V(x))/V_p ) * dV(x)/dx
        
        if plot:
            plt.plot(V_d,I * 1e6)
            plt.legend([f'$V_g = {V_g}$' for V_g in V_g_values])
            
            plt.xlabel(r'$V_d (V)$')
            plt.ylabel(r'$I_{sd} (\mu A)$')
            plt.grid(True)
            
            plt.show()
        # return J
        pass

    def diffusion_1D_conductivity(self):
        q       = self.q
        mu_p    = self.hole_mobility
        p0      = self.hole_density
        W      = self.W
        T      = self.T
        L      = self.L
        
        G = q*mu_p*p0*W*T/L
        return(G)       

    def diffusion_1D_current(self, V_d):
        q       = self.q
        mu_p    = self.hole_mobility
        p0      = self.hole_density
        V_g      = self.V_G
        V_p      = self.V_p
        # V_d      = V_d
        W      = self.W
        T      = self.T
        L      = self.L
        G       = self.G
        
        I = np.zeros_like(V_d)
        I0 = G * ( 1 - (V_g - .5*V_d)/V_p ) * V_d
        I1 = G * ( V_d - (V_g**2)/(2*V_p) )
        I[V_d<V_g] = I0[V_d<V_g]
        I[V_d>V_g] = I1[V_d>V_g]
        
        return(I)
          
    
my_oect = OECT_model()
my_oect.diffusion_1D()