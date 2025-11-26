"""
dashboard_advanced.py - VERSIÓN ULTRA-COMPATIBLE
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Cargar datos
df = pd.read_csv("diabetes_dataset_mini.csv")

# App simple pero completa
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1("🏥 Diabetes Analytics Dashboard", className="text-center my-4"),
    
    # Filtros
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filtros"),
                dbc.CardBody([
                    html.Label("Edad:"),
                    dcc.RangeSlider(
                        id='age-slider',
                        min=20, max=80, step=1,
                        value=[25, 65],
                        marks={i: str(i) for i in range(20, 81, 10)}
                    ),
                    
                    html.Label("Glucosa:"),
                    dcc.RangeSlider(
                        id='glucose-slider', 
                        min=50, max=250, step=5,
                        value=[80, 180],
                        marks={50: '50', 100: '100', 150: '150', 200: '200', 250: '250'}
                    ),
                    
                    html.Label("Resultado:"),
                    dcc.RadioItems(
                        id='outcome-filter',
                        options=[
                            {'label': 'Todos', 'value': 'all'},
                            {'label': 'Solo Positivos', 'value': 1},
                            {'label': 'Solo Negativos', 'value': 0}
                        ],
                        value='all',
                        labelStyle={'display': 'block'}
                    )
                ])
            ])
        ], md=4),
        
        # Gráficos
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='outcome-pie'), md=6),
                dbc.Col(dcc.Graph(id='glucose-hist'), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='scatter-plot'), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='correlation-heatmap'), md=12),
            ])
        ], md=8)
    ])
], fluid=True)

@app.callback(
    [Output('outcome-pie', 'figure'),
     Output('glucose-hist', 'figure'), 
     Output('scatter-plot', 'figure'),
     Output('correlation-heatmap', 'figure')],
    [Input('age-slider', 'value'),
     Input('glucose-slider', 'value'),
     Input('outcome-filter', 'value')]
)
def update_dashboard(age_range, glucose_range, outcome_filter):
    # Filtrar datos
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Glucose'] >= glucose_range[0]) & (df['Glucose'] <= glucose_range[1])
    ]
    
    if outcome_filter != 'all':
        filtered_df = filtered_df[filtered_df['Outcome'] == outcome_filter]
    
    # 1. Gráfico de torta
    outcome_counts = filtered_df['Outcome'].value_counts()
    pie_fig = px.pie(
        values=outcome_counts.values,
        names=['Negativo', 'Positivo'],
        title='Distribución de Diabetes'
    )
    
    # 2. Histograma
    hist_fig = px.histogram(
        filtered_df, 
        x='Glucose',
        color=filtered_df['Outcome'].map({0: 'Negativo', 1: 'Positivo'}),
        title='Distribución de Glucosa',
        barmode='overlay'
    )
    
    # 3. Scatter plot
    scatter_fig = px.scatter(
        filtered_df,
        x='Age',
        y='Glucose', 
        color=filtered_df['Outcome'].map({0: 'Negativo', 1: 'Positivo'}),
        size='BMI',
        title='Edad vs Glucosa',
        hover_data=['BloodPressure']
    )
    
    # 4. Mapa de correlación
    corr_matrix = filtered_df.corr()
    heatmap_fig = px.imshow(
        corr_matrix,
        title='Mapa de Correlación',
        color_continuous_scale='RdBu'
    )
    
    return pie_fig, hist_fig, scatter_fig, heatmap_fig

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)
