"""
dashboard_advanced_sophisticated.py
Dashboard super avanzado con múltiples funciones
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime
import scipy.stats as stats

# Cargar datos
df = pd.read_csv("diabetes_dataset_mini.csv")

# ========== CONFIGURACIÓN AVANZADA ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "🏥 Medical Analytics Dashboard"

# ========== LAYOUT SUPER MEJORADO ==========
app.layout = dbc.Container([
    # HEADER CON LOGO Y FECHA
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🏥 MEDICAL ANALYTICS DASHBOARD", 
                       className="text-center text-white mb-1",
                       style={'fontWeight': 'bold', 'fontSize': '2.5rem'}),
                html.P("Sistema de Monitoreo y Predicción de Diabetes", 
                      className="text-center text-light mb-4",
                      style={'fontSize': '1.2rem'})
            ])
        ], width=12)
    ], className="bg-primary rounded-top p-3"),
    
    # ALERTA DE ACTUALIZACIÓN EN TIEMPO REAL
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.Span("🔄 Actualizado: "),
                html.Span(id="live-update-time", className="fw-bold")
            ], color="info", className="text-center py-2")
        ], width=12)
    ]),
    
    # PRIMERA FILA: KPI CARDS CON ANIMACIONES
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div("👥", className="text-center fs-1 mb-2"),
                html.H4(id="total-patients", className="card-title text-center"),
                html.P("Total Pacientes", className="card-text text-center"),
                dbc.Progress(id="progress-total", value=100, color="primary", className="mt-2")
            ])
        ], color="primary", inverse=True), md=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div("🩺", className="text-center fs-1 mb-2"),
                html.H4(id="diabetes-cases", className="card-title text-center"),
                html.P("Casos Diabetes", className="card-text text-center"),
                dbc.Progress(id="progress-cases", value=45, color="danger", className="mt-2")
            ])
        ], color="danger", inverse=True), md=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div("📊", className="text-center fs-1 mb-2"),
                html.H4(id="diabetes-rate", className="card-title text-center"),
                html.P("Tasa Incidencia", className="card-text text-center"),
                html.Div(id="trend-indicator", className="text-center")
            ])
        ], color="warning", inverse=True), md=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div("📈", className="text-center fs-1 mb-2"),
                html.H4(id="avg-glucose", className="card-title text-center"),
                html.P("Glucosa Promedio", className="card-text text-center"),
                html.Small(id="glucose-status", className="text-center d-block")
            ])
        ], color="info", inverse=True), md=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div("⚖️", className="text-center fs-1 mb-2"),
                html.H4(id="avg-bmi", className="card-title text-center"),
                html.P("IMC Promedio", className="card-text text-center"),
                html.Small(id="bmi-status", className="text-center d-block")
            ])
        ], color="success", inverse=True), md=2),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div("🎯", className="text-center fs-1 mb-2"),
                html.H4(id="risk-score", className="card-title text-center"),
                html.P("Score de Riesgo", className="card-text text-center"),
                dbc.Progress(id="progress-risk", value=65, color="warning", className="mt-2")
            ])
        ], color="secondary", inverse=True), md=2),
    ], className="mb-4"),
    
    # SEGUNDA FILA: FILTROS AVANZADOS
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎛️ Panel de Control Avanzado", className="bg-dark text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("👤 Rango de Edad:", className="fw-bold"),
                            dcc.RangeSlider(
                                id='age-slider',
                                min=20, max=80, step=1,
                                value=[25, 65],
                                marks={i: str(i) for i in range(20, 81, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("🩸 Nivel de Glucosa:", className="fw-bold"),
                            dcc.RangeSlider(
                                id='glucose-slider',
                                min=50, max=250, step=5,
                                value=[80, 180],
                                marks={50: '50', 100: '100', 150: '150', 200: '200', 250: '250'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("⚖️ Índice de Masa Corporal:", className="fw-bold"),
                            dcc.RangeSlider(
                                id='bmi-slider',
                                min=15, max=50, step=0.5,
                                value=[20, 35],
                                marks={15: '15', 25: '25', 35: '35', 50: '50'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("🎯 Estado de Diabetes:", className="fw-bold"),
                            dcc.Dropdown(
                                id='outcome-filter',
                                options=[
                                    {'label': '🔵 Todos los casos', 'value': 'all'},
                                    {'label': '🟢 Solo negativos', 'value': 0},
                                    {'label': '🔴 Solo positivos', 'value': 1}
                                ],
                                value='all',
                                clearable=False
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("📋 Agrupar por:", className="fw-bold"),
                            dcc.Dropdown(
                                id='group-by',
                                options=[
                                    {'label': 'Edad', 'value': 'Age'},
                                    {'label': 'Resultado', 'value': 'Outcome'},
                                    {'label': 'Grupo de Glucosa', 'value': 'Glucose_Group'}
                                ],
                                value='Outcome',
                                clearable=False
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("🎨 Tema de Color:", className="fw-bold"),
                            dcc.Dropdown(
                                id='color-theme',
                                options=[
                                    {'label': '🔵 Azul', 'value': 'blues'},
                                    {'label': '🔴 Rojo', 'value': 'reds'},
                                    {'label': '🟢 Verde', 'value': 'greens'},
                                    {'label': '🟣 Púrpura', 'value': 'purples'}
                                ],
                                value='blues',
                                clearable=False
                            )
                        ], md=4),
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("🔄 Actualizar Dashboard", id="update-btn", color="primary", size="lg", className="w-100")
                        ], md=6),
                        dbc.Col([
                            dbc.Button("📊 Generar Reporte", id="report-btn", color="success", size="lg", className="w-100")
                        ], md=6),
                    ], className="mt-3")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # TERCERA FILA: GRÁFICOS PRINCIPALES
    dbc.Row([
        # Gráfico 1: Distribución con subplots
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Distribución de Pacientes", className="bg-dark text-white"),
                dbc.CardBody(dcc.Graph(id="distribution-plot"))
            ])
        ], md=6),
        
        # Gráfico 2: Correlación avanzada
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🕸️ Mapa de Correlación Avanzado", className="bg-dark text-white"),
                dbc.CardBody(dcc.Graph(id="correlation-plot"))
            ])
        ], md=6),
    ], className="mb-4"),
    
    # CUARTA FILA: GRÁFICOS SECUNDARIOS
    dbc.Row([
        # Gráfico 3: Scatter matrix
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 Matriz de Dispersión", className="bg-dark text-white"),
                dbc.CardBody(dcc.Graph(id="scatter-matrix"))
            ])
        ], md=6),
        
        # Gráfico 4: Box plots avanzados
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📦 Análisis de Distribución", className="bg-dark text-white"),
                dbc.CardBody(dcc.Graph(id="box-plot"))
            ])
        ], md=6),
    ], className="mb-4"),
    
    # QUINTA FILA: ANÁLISIS ESTADÍSTICO
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Análisis Estadístico Avanzado", className="bg-dark text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Div(id="stats-summary"), md=6),
                        dbc.Col(dcc.Graph(id="probability-plot"), md=6),
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # SEXTA FILA: TABLA INTERACTIVA
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Datos en Tiempo Real", className="bg-dark text-white"),
                dbc.CardBody([
                    html.Div(id="data-table"),
                    dbc.Pagination(id="pagination", max_value=5, active_page=1, className="mt-3")
                ])
            ])
        ], width=12)
    ]),
    
    # FOOTER
    dbc.Row([
        dbc.Col([
            html.Footer([
                html.Hr(),
                html.P("🏥 Medical Analytics Dashboard v2.0 | © 2024 Sistema de Monitoreo de Diabetes", 
                      className="text-center text-muted")
            ])
        ], width=12)
    ], className="mt-4")
], fluid=True, style={'backgroundColor': '#1a1a1a'})

# ========== CALLBACKS AVANZADOS ==========
@app.callback(
    [Output("live-update-time", "children"),
     Output("total-patients", "children"),
     Output("diabetes-cases", "children"),
     Output("diabetes-rate", "children"),
     Output("avg-glucose", "children"),
     Output("avg-bmi", "children"),
     Output("risk-score", "children"),
     Output("progress-cases", "value"),
     Output("progress-risk", "value"),
     Output("trend-indicator", "children"),
     Output("glucose-status", "children"),
     Output("bmi-status", "children")],
    [Input("update-btn", "n_clicks"),
     Input("age-slider", "value"),
     Input("glucose-slider", "value"),
     Input("bmi-slider", "value"),
     Input("outcome-filter", "value")],
    prevent_initial_call=False
)
def update_kpis(n_clicks, age_range, glucose_range, bmi_range, outcome_filter):
    # Filtrar datos
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Glucose'] >= glucose_range[0]) & (df['Glucose'] <= glucose_range[1]) &
        (df['BMI'] >= bmi_range[0]) & (df['BMI'] <= bmi_range[1])
    ]
    
    if outcome_filter != 'all':
        filtered_df = filtered_df[filtered_df['Outcome'] == outcome_filter]
    
    # Calcular métricas
    total_patients = len(filtered_df)
    diabetes_cases = filtered_df['Outcome'].sum()
    diabetes_rate = (diabetes_cases / total_patients * 100) if total_patients > 0 else 0
    avg_glucose = filtered_df['Glucose'].mean()
    avg_bmi = filtered_df['BMI'].mean()
    
    # Score de riesgo (algoritmo simple)
    risk_factors = 0
    if avg_glucose > 140: risk_factors += 1
    if avg_bmi > 30: risk_factors += 1
    if diabetes_rate > 50: risk_factors += 1
    risk_score = (risk_factors / 3) * 100
    
    # Indicadores de tendencia
    trend = "🟢 Estable" if diabetes_rate < 30 else "🟡 Moderado" if diabetes_rate < 60 else "🔴 Alto"
    glucose_status = "🟢 Normal" if avg_glucose < 100 else "🟡 Elevado" if avg_glucose < 126 else "🔴 Peligroso"
    bmi_status = "🟢 Saludable" if avg_bmi < 25 else "🟡 Sobrepeso" if avg_bmi < 30 else "🔴 Obesidad"
    
    return (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{total_patients}",
        f"{diabetes_cases}",
        f"{diabetes_rate:.1f}%",
        f"{avg_glucose:.1f}",
        f"{avg_bmi:.1f}",
        f"{risk_score:.0f}%",
        diabetes_rate,
        risk_score,
        trend,
        glucose_status,
        bmi_status
    )

@app.callback(
    [Output("distribution-plot", "figure"),
     Output("correlation-plot", "figure"),
     Output("scatter-matrix", "figure"),
     Output("box-plot", "figure"),
     Output("probability-plot", "figure")],
    [Input("update-btn", "n_clicks")],
    [State("age-slider", "value"),
     State("glucose-slider", "value"),
     State("bmi-slider", "value"),
     State("outcome-filter", "value"),
     State("group-by", "value"),
     State("color-theme", "value")]
)
def update_advanced_charts(n_clicks, age_range, glucose_range, bmi_range, outcome_filter, group_by, color_theme):
    # Filtrar datos
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Glucose'] >= glucose_range[0]) & (df['Glucose'] <= glucose_range[1]) &
        (df['BMI'] >= bmi_range[0]) & (df['BMI'] <= bmi_range[1])
    ]
    
    if outcome_filter != 'all':
        filtered_df = filtered_df[filtered_df['Outcome'] == outcome_filter]
    
    # 1. GRÁFICO DE DISTRIBUCIÓN MEJORADO
    dist_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribución por Edad', 'Distribución por Glucosa', 
                       'Distribución por IMC', 'Distribución por Presión'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    dist_fig.add_trace(go.Histogram(x=filtered_df['Age'], name='Edad', nbinsx=20), row=1, col=1)
    dist_fig.add_trace(go.Histogram(x=filtered_df['Glucose'], name='Glucosa', nbinsx=20), row=1, col=2)
    dist_fig.add_trace(go.Histogram(x=filtered_df['BMI'], name='IMC', nbinsx=20), row=2, col=1)
    dist_fig.add_trace(go.Histogram(x=filtered_df['BloodPressure'], name='Presión', nbinsx=20), row=2, col=2)
    
    dist_fig.update_layout(height=600, showlegend=False, template="plotly_dark")
    
    # 2. MAPA DE CORRELACIÓN 3D
    corr_matrix = filtered_df.corr()
    corr_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=color_theme,
        hoverongaps=False,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    corr_fig.update_layout(
        title="Matriz de Correlación Avanzada",
        height=500,
        template="plotly_dark"
    )
    
    # 3. MATRIZ DE DISPERSIÓN
    scatter_fig = px.scatter_matrix(
        filtered_df,
        dimensions=['Age', 'Glucose', 'BMI', 'BloodPressure'],
        color='Outcome',
        title="Matriz de Dispersión Multivariable",
        template="plotly_dark"
    )
    scatter_fig.update_layout(height=500)
    
    # 4. BOX PLOTS AVANZADOS
    box_fig = go.Figure()
    
    variables = ['Glucose', 'BMI', 'BloodPressure', 'Age']
    for i, var in enumerate(variables):
        box_fig.add_trace(go.Box(
            y=filtered_df[var],
            name=var,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    box_fig.update_layout(
        title="Distribución de Variables Clínicas",
        height=500,
        template="plotly_dark",
        showlegend=False
    )
    
    # 5. GRÁFICO DE PROBABILIDAD
    prob_fig = px.histogram(
        filtered_df, 
        x="Glucose", 
        color="Outcome", 
        marginal="box",
        title="Distribución de Glucosa por Resultado",
        template="plotly_dark",
        barmode="overlay",
        opacity=0.7
    )
    prob_fig.update_layout(height=400)
    
    return dist_fig, corr_fig, scatter_fig, box_fig, prob_fig

@app.callback(
    [Output("stats-summary", "children"),
     Output("data-table", "children")],
    [Input("update-btn", "n_clicks")],
    [State("age-slider", "value"),
     State("glucose-slider", "value"),
     State("bmi-slider", "value")]
)
def update_stats_and_table(n_clicks, age_range, glucose_range, bmi_range):
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Glucose'] >= glucose_range[0]) & (df['Glucose'] <= glucose_range[1]) &
        (df['BMI'] >= bmi_range[0]) & (df['BMI'] <= bmi_range[1])
    ]
    
    # Estadísticas avanzadas
    stats_text = [
        html.H5("📈 Resumen Estadístico", className="text-white"),
        html.P(f"📊 Total de registros: {len(filtered_df)}", className="text-light"),
        html.P(f"🎯 Media de glucosa: {filtered_df['Glucose'].mean():.2f}", className="text-light"),
        html.P(f"📏 Desviación estándar: {filtered_df['Glucose'].std():.2f}", className="text-light"),
        html.P(f"📈 Correlación glucosa-IMC: {filtered_df['Glucose'].corr(filtered_df['BMI']):.3f}", className="text-light"),
        html.P(f"🩺 Tasa de diabetes: {(filtered_df['Outcome'].mean()*100):.1f}%", className="text-light"),
    ]
    
    # Tabla de datos
    table = dbc.Table.from_dataframe(
        filtered_df.head(10).round(2),
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="text-white"
    )
    
    return stats_text, table

# ========== EJECUCIÓN ==========
# === CONFIGURACIÓN PARA RAILWAY ===
if __name__ == '__main__':
    # Para desarrollo local
    debug_mode = os.environ.get('RAILWAY_ENVIRONMENT') != 'production'
    port = int(os.environ.get("PORT", 8050))
    
    print("🚀 Iniciando servidor Dash...")
    print(f"🌐 Entorno: {'Producción' if not debug_mode else 'Desarrollo'}")
    print(f"🔗 Puerto: {port}")
    print(f"📊 URL: http://0.0.0.0:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )