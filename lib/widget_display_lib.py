# from sympy import *
# import ipywidgets
from spb import plot,plot_implicit

def make_widget(domain, func, params, MPL_kwargs=None):
    '''Makes a ipython widget with sliders for the indicated ranges'''

    options = dict( show=False, params=params, **MPL_kwargs )

    plot_type = 'plot'
    if 'plot_type' in options.keys():
        plot_type = options['plot_type']

    if plot_type == 'plot':
        p = plot(
            func, domain,
            nb_of_points=1000,
            # xlim=MPL_kwargs['xlim'],
            **options,
            # **MPL_kwargs,
            # **kwargs,
            )
    elif plot_type == 'implicit':
        p = plot_implicit(
            func, domain,
            nb_of_points=1000,
            # xlim=MPL_kwargs['xlim'],
            **options,
            # **MPL_kwargs,
            # **kwargs,
            )

    return p