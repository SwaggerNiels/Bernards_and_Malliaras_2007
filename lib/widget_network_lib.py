import os
import html
from pyvis.network import Network
from IPython.display import HTML, IFrame

def make_network():
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # Add nodes
    net.add_node(1, label="Node 1", size=20, color="#FF0000")
    net.add_node(2, label="Node 2", size=15, color="#00FF00")
    net.add_node(3, label="Node 3", size=10, color="#0000FF")

    # Add edges
    net.add_edge(1, 2, color="#FFFF00")
    net.add_edge(2, 3, color="#00FFFF")
    net.add_edge(3, 1, color="#FF00FF")

    return net


def show_network(net):
    # Add physics options (optional)
    net.set_options(
        """
    var options = {
    "physics": {
        "enabled": true,
        "barnesHut": {
        "gravitationalConstant": -80000,
        "centralGravity": 0.1,
        "springLength": 100,
        "springConstant": 0.04,
        "damping": 0.09,
        "avoidOverlap": 0
        },
        "minVelocity": 0.75
    }
    }
    """
    )


    def get_network_html_code(nt: Network) -> str:
        fname = "temp_file.html"
        nt.save_graph(fname)

        with open(fname, "r") as f:
            html_test = f.read()

        os.remove(fname)
        return html_test


    html_content = get_network_html_code(net)

    iframe = f"""<iframe srcdoc="{html.escape(html_content)}" width=1000 height=500></iframe>"""

    return iframe

# net = make_network()
# iframe = show_network(net)
# HTML(iframe)