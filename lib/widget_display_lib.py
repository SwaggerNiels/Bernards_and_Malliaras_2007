# from sympy import *
# import ipywidgets
from spb import plot

def make_widget(domain, func, params, MPL_kwargs=None):
    '''Makes a ipython widget with sliders for the indicated ranges'''

    options = dict( show=False, params=params, **MPL_kwargs )

    p = plot(
        func, domain,
        nb_of_points=500,
        # xlim=MPL_kwargs['xlim'],
        **options,
        # **MPL_kwargs,
        # **kwargs,
        )

    return p