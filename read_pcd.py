import os
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data')
data_file = os.path.join(data_dir, 'Cropped_point_cloud.pcd')
assert os.path.isfile(data_file)

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Ingot Point Cloud'),
    dcc.Slider(min=1,
               max=10,
               marks={i: f'{i*0.1}' for i in range(1, 10)},
               value=5,
               id="marker-size",
               ),
    dcc.Graph(id="pcd"),
])


@app.callback(Output("pcd", "figure"),
              Input("marker-size", "value"))
def display_pcd(value):
    pc = o3d.io.read_point_cloud(data_file)
    data = np.array(pc.points, dtype='float32')
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=data[:, 0],
                               y=data[:, 1],
                               z=data[:, 2],
                               mode='markers',
                               marker_color=data[:, 2],
                               marker={'size': float(value * 0.1)}))
    return fig


app.run_server(debug=True)