import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Configurazioni iniziali
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Progetto data manipulation & visualization - Alessandro Attino",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stili CSS
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .main {
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #1f2937;
        }
        .stMarkdown {
            color: #4b5563;
        }
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_and_clean_data():
    """Carica e prepara i dati con cache"""
    try:
        # Caricamento dati
        apps_df = pd.read_csv('googleplaystore.csv')
        reviews_df = pd.read_csv('googleplaystore_user_reviews.csv')
        
        # Pulizia base - rimuovi app con categoria mal formattata e duplicati
        apps_df = apps_df[apps_df['Category'] != '1.9'].copy()
        apps_df = apps_df.drop_duplicates(subset=['App'], keep='first')  # Aggiungi questa riga
        
        # Conversioni di base
        apps_df['Size_MB'] = apps_df['Size'].apply(lambda x: 
            float(str(x).replace('M', '')) if isinstance(x, str) and 'M' in x
            else float(str(x).replace('k', '')) / 1024 if isinstance(x, str) and 'k' in x
            else np.nan)
        
        apps_df['Installs_Clean'] = apps_df['Installs'].str.replace('+', '').str.replace(',', '').astype(float)
        apps_df['Price_Clean'] = apps_df['Price'].apply(lambda x: 
            float(str(x).replace('$', '')) if x != 'Free' and x != '0' else 0.0)
        
        # Conversioni aggiuntive
        apps_df['Rating'] = pd.to_numeric(apps_df['Rating'], errors='coerce')
        apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'])
        apps_df['Log_Installs'] = np.log1p(apps_df['Installs_Clean'])
        apps_df['Log_Size'] = np.log1p(apps_df['Size_MB'])
        apps_df['Days_Since_Update'] = (pd.Timestamp.now() - apps_df['Last Updated']).dt.days
        
        # Pulizia versione Android
        apps_df['Android_Ver_Clean'] = apps_df['Android Ver'].str.extract(r'(\d+\.?\d?)').astype(float)
        
        return apps_df, reviews_df
        
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati: {str(e)}")
        raise

# Caricamento dati
apps_df, reviews_df = load_and_clean_data()

# Header principale
st.title("Progetto data manipulation & visualization - Alessandro Attino")

# Filtri nella sidebar
with st.sidebar:
    st.title("Filtri analisi")

    # Filtro categorie semplificato
    st.markdown("### Selezione categorie")
    selected_categories = st.multiselect(
        "Seleziona una o più categorie",
        options=sorted(apps_df['Category'].unique()),
        default=None,
        help="Lascia vuoto per visualizzare tutte le categorie"
    )

    # Filtro prezzo
    st.markdown("### Filtro prezzo")
    price_filter = st.radio(
        "Tipo di app",
        options=["Tutte le app", "Solo app gratuite", "Solo app a pagamento"],
        index=0
    )

# Applicazione filtri
filtered_df = apps_df.copy()
if selected_categories:
    filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
if price_filter == "Solo app gratuite":
    filtered_df = filtered_df[filtered_df['Price_Clean'] == 0]
elif price_filter == "Solo app a pagamento":
    filtered_df = filtered_df[filtered_df['Price_Clean'] > 0]

# Statistiche generali in alto
metrics_columns = st.columns(4)
with metrics_columns[0]:
    st.metric("Totale app", f"{len(filtered_df):,}")
with metrics_columns[1]:
    st.metric("Rating medio", f"{filtered_df['Rating'].mean():.2f}")
with metrics_columns[2]:
    st.metric("% app gratuite", f"{(filtered_df['Price_Clean'] == 0).mean()*100:.1f}%")
with metrics_columns[3]:
    st.metric("Dimensione media", f"{filtered_df['Size_MB'].mean():.1f} MB")

# Tabs principali
tabs = st.tabs(["Overview di mercato", "Analisi correlazioni", "Analisi temporale", "Analisi tecnica"])

with tabs[0]:
    st.header("Overview di mercato")
    
    # 1. Distribuzione delle app per categoria e rating medio
    st.subheader("Distribuzione delle app per categoria e rating medio")
    
    @st.cache_data
    def create_category_distribution(df):
        """Crea un grafico a barre che replica esattamente quello originale"""
        # Calcola le statistiche per categoria
        category_stats = df.groupby('Category').agg({
            'App': 'count',
            'Rating': 'mean'
        }).reset_index()
        
        # Creazione grafico con plotly express - usiamo ESATTAMENTE gli stessi parametri del codice originale
        fig = px.bar(
            category_stats,
            x='Category',
            y='App',
            color='Rating',
            title='Distribuzione delle app per categoria e rating medio',
            labels={
                'App': 'Numero di app',
                'Category': 'Categoria',
                'Rating': 'Rating medio'
            },
            color_continuous_scale=[[0, '#B30000'], [0.4, '#FF0000'],
                                [0.6, '#FFA500'], [0.75, '#2ECC40'],
                                [1, '#00B300']],
            range_color=[3.2, 4.8]
        )
        
        # Aggiorniamo il layout per essere il più fedele possibile all'originale
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=True,
            height=600,
            title_x=0.5,
            font=dict(family="Arial", size=12),
            margin=dict(t=100, l=50, r=50, b=100),
            plot_bgcolor='rgba(240, 245, 255, 0.95)'  # Sfondo azzurrino chiaro come nell'originale
        )
        
        return fig
    
    st.plotly_chart(create_category_distribution(filtered_df), use_container_width=True)
    
    # 2. Distribuzione dei prezzi delle app a pagamento
    st.subheader("Distribuzione dei prezzi delle app a pagamento")
    
    @st.cache_data
    def create_price_distribution(df):
        df_paid = df[df['Price_Clean'] > 0].copy()
        
        if len(df_paid) == 0:
            st.warning("Nessuna app a pagamento trovata con i filtri selezionati")
            return None, None
            
        price_ranges = [0, 1, 2.99, 4.99, 9.99, float('inf')]
        price_labels = ['0-1$', '1-2.99$', '3-4.99$', '5-9.99$', '10$+']
        
        df_paid['price_range'] = pd.cut(df_paid['Price_Clean'],
                                    bins=price_ranges,
                                    labels=price_labels)
        
        price_distribution = df_paid.groupby('price_range').agg({
            'App': 'count',
            'Rating': 'mean'
        }).reset_index()
        
        price_distribution['percentage'] = (
            price_distribution['App'] / len(df_paid)
        ) * 100
        
        fig = go.Figure(data=[
            go.Bar(
                x=price_distribution['price_range'],
                y=price_distribution['percentage'],
                marker=dict(
                    color=price_distribution['Rating'],
                    colorscale=[
                        [0, '#B30000'], 
                        [0.4, '#FF0000'], 
                        [0.6, '#FFA500'], 
                        [0.75, '#2ECC40'], 
                        [1, '#00B300']
                    ],
                    colorbar=dict(title="Rating medio")
                ),
                text=price_distribution['percentage'].round(1).astype(str) + '%',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            height=500,
            margin=dict(t=20, l=50, r=50, b=50),
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            xaxis=dict(
                title='Fascia di prezzo',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Percentuale di app (%)',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey'
            )
        )
        
        return fig, df_paid

    fig_prices, df_paid = create_price_distribution(filtered_df)
    if fig_prices:
        st.plotly_chart(fig_prices, use_container_width=True)
        nan_ranges = df_paid[df_paid['Rating'].isna()]['price_range'].unique()
        if len(nan_ranges) > 0:
            st.caption("Nota: Le barre grigie indicano fasce di prezzo per cui non sono disponibili valutazioni")
   
    # 3. Mappa competitiva del mercato 
    st.subheader("Mappa competitiva del mercato")
    
    @st.cache_data
    def create_market_competition(df):
        """Crea una mappa competitiva del mercato basata sul numero di app e rating"""
        market_analysis = df.groupby('Category').agg({
            'App': 'count',
            'Rating': 'mean',
            'Price_Clean': lambda x: (x > 0).mean() * 100,
            'Installs_Clean': 'mean'
        }).round(2)

        market_analysis.columns = ['num_apps', 'avg_rating', 'paid_perc', 'avg_installs']

        # Calcolo indice dimensione mercato
        market_analysis['market_size_index'] = (
            (market_analysis['num_apps'] - market_analysis['num_apps'].min()) /
            (market_analysis['num_apps'].max() - market_analysis['num_apps'].min())
        )

        # Calcolo opportunity score
        market_analysis['opportunity_score'] = (
            market_analysis['avg_rating'] * 0.4 +
            (1 - market_analysis['market_size_index']) * 0.3 +
            market_analysis['paid_perc'] * 0.3
        )

        # Creazione grafico
        fig = go.Figure(data=[
            go.Scatter(
                x=market_analysis['num_apps'],
                y=market_analysis['avg_rating'],
                mode='markers+text',
                text=market_analysis.index,
                textposition='top center',
                marker=dict(
                    size=market_analysis['market_size_index'] * 50 + 20,
                    color=market_analysis['avg_rating'],
                    colorscale=[[0, '#B30000'], [0.4, '#FF0000'],
                               [0.6, '#FFA500'], [0.75, '#2ECC40'],
                               [1, '#00B300']],
                    colorbar=dict(
                        title="Rating medio"
                    ),
                    cmin=3.2,
                    cmax=4.8
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Numero app: %{x}<br>" +
                    "Rating medio: %{marker.color:.2f}<br>" +
                    "Propensione al pagamento: %{marker.size:.1f}%<br>" +
                    "<extra></extra>"
                )
            )
        ])

        # Aggiungi annotazione con migliori opportunità
        top_opportunities = market_analysis.nlargest(5, 'opportunity_score')
        top_text = "<b>Top 5 opportunità di mercato:</b><br>"
        for cat in top_opportunities.index:
            score = market_analysis.loc[cat, 'opportunity_score']
            rating = market_analysis.loc[cat, 'avg_rating']
            apps = market_analysis.loc[cat, 'num_apps']
            paid = market_analysis.loc[cat, 'paid_perc']
            top_text += f"<b>{cat}</b>: score {score:.2f} (rating {rating:.1f}, app {apps}, a pagamento {paid:.1f}%)<br>"

        fig.add_annotation(
            x=0.99,  
            y=0.01,  
            xref="paper",
            yref="paper",
            xanchor="right",  
            yanchor="bottom",  
            text=top_text,
            showarrow=False,
            font=dict(size=11),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            borderpad=6
        )

        fig.update_layout(
            title='Mappa competitiva del mercato',
            xaxis_title='Numero di app (competizione)',
            yaxis_title='Rating medio',
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            font=dict(family="Arial", size=12),
            margin=dict(t=50, l=50, r=50, b=50)
        )
        
        return fig
    
    st.plotly_chart(create_market_competition(filtered_df), use_container_width=True)
    
    # 4. Struttura del mercato per categoria
    st.subheader("Struttura del mercato per categoria")
    
    @st.cache_data
    def create_market_structure(df):
        market_stats = df.groupby('Category').agg({
            'Installs_Clean': 'sum',
            'Rating': 'mean',
            'App': 'count',
            'Price_Clean': lambda x: (x > 0).mean()
        }).reset_index()
        
        total_market = market_stats['Installs_Clean'].sum()
        market_stats['concentration'] = market_stats['Installs_Clean'] / total_market
        
        # Calcolo indice di stabilità
        stabilities = []
        for category in market_stats['Category']:
            cat_data = df[df['Category'] == category]
            rating_stability = 1 - (cat_data['Rating'].std() / cat_data['Rating'].mean() if cat_data['Rating'].mean() > 0 else 0)
            update_freq = cat_data['Days_Since_Update'].std() / 365
            stability = (rating_stability + (1 - update_freq)) / 2
            stabilities.append(stability)
        
        market_stats['stability'] = stabilities
        market_stats['size'] = (market_stats['Price_Clean'] * 50) + 20
        
        fig = go.Figure(data=[
            go.Scatter(
                x=market_stats['concentration'],
                y=market_stats['stability'],
                mode='markers+text',
                text=market_stats['Category'],
                textposition='top center',
                marker=dict(
                    size=market_stats['size'],
                    color=market_stats['Rating'],
                    colorscale=[
                        [0, '#B30000'], 
                        [0.4, '#FF0000'], 
                        [0.6, '#FFA500'], 
                        [0.75, '#2ECC40'], 
                        [1, '#00B300']
                    ],
                    colorbar=dict(title='Rating medio')
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Concentrazione: %{x:.3f}<br>" +
                    "Stabilità: %{y:.3f}<br>" +
                    "Rating medio: %{marker.color:.2f}<br>" +
                    "<extra></extra>"
                )
            )
        ])
        
        fig.update_layout(
            height=600,
            margin=dict(t=20, l=50, r=50, b=50),
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            xaxis=dict(
                title='Indice di concentrazione del mercato',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Indice di stabilità',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey'
            )
        )
        
        return fig
    
    st.plotly_chart(create_market_structure(filtered_df), use_container_width=True)

with tabs[1]:
    st.header("Analisi correlazioni")
    
    # 1. Correlazioni Pearson vs Spearman
    st.subheader("Confronto correlazioni Pearson vs Spearman")
    
    @st.cache_data
    def create_correlation_matrices(df):
        metrics = {
            'Rating': 'Rating',
            'Price_Clean': 'Prezzo',
            'Log_Installs': 'Log installazioni',
            'Log_Size': 'Log dimensione',
            'Days_Since_Update': 'Giorni da ultimo aggiornamento'
        }
        
        corr_data = df[metrics.keys()]
        pearson_corr = corr_data.corr(method='pearson').round(3)
        spearman_corr = corr_data.corr(method='spearman').round(3)
        
        for corr_matrix in [pearson_corr, spearman_corr]:
            corr_matrix.columns = metrics.values()
            corr_matrix.index = metrics.values()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Correlazioni di Pearson', 'Correlazioni di Spearman'),
            horizontal_spacing=0.2
        )
        
        for i, (corr, pos) in enumerate([(pearson_corr, 1), (spearman_corr, 2)]):
            fig.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=corr.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ),
                row=1, col=pos
            )
        
        fig.update_layout(
            height=700,
            margin=dict(t=100, l=100, r=100, b=100),
            font=dict(family="Arial", size=12),
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    st.plotly_chart(create_correlation_matrices(filtered_df), use_container_width=True)
    
    # 2. Scatter plots delle relazioni principali
    st.subheader("Scatter plots delle relazioni principali")
    
    @st.cache_data
    def create_scatter_matrix(df):
        # Preparazione dati
        df = df.copy()
        df['Last Updated'] = pd.to_datetime(df['Last Updated'])
        max_date = df['Last Updated'].max()
        df['Days_Since_Update'] = (max_date - df['Last Updated']).dt.days
        
        scatter_pairs = [
            ('Rating', 'Log_Installs'),
            ('Rating', 'Log_Size'),
            ('Log_Size', 'Log_Installs'),
            ('Rating', 'Days_Since_Update')
        ]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'{x.replace("_", " ")} vs {y.replace("_", " ")}'
                for x, y in scatter_pairs
            ],
            horizontal_spacing=0.15,
            vertical_spacing=0.15
        )

        colorscale = [
            [0, 'red'],       # Rating 1
            [0.25, 'orange'], # Rating 2
            [0.5, 'yellow'],  # Rating 3
            [0.75, 'lightgreen'], # Rating 4
            [1, 'green']      # Rating 5
        ]

        for idx, (x, y) in enumerate(scatter_pairs):
            row = idx // 2 + 1
            col = idx % 2 + 1

            if y == 'Days_Since_Update':
                # Aggiunta jitter per evitare sovrapposizioni
                jittered_x = df[x] + np.random.normal(0, 0.05, len(df))

                # Calcolo media mobile
                rating_range = np.arange(1, 5.1, 0.1)
                days_mean = []
                for r in rating_range:
                    mask = (df[x] >= r - 0.2) & (df[x] < r + 0.2)
                    mean_val = df.loc[mask, y].mean()
                    days_mean.append(mean_val)

                # Creazione del testo hover
                hover_text = [
                    f"App: {app}<br>" +
                    f"Categoria: {cat}<br>" +
                    f"Rating: {rating:.2f}<br>" +
                    f"Giorni da ultimo aggiornamento: {days:.0f}<br>" +
                    f"Rating: {rating:.1f}<br>" +
                    f"Prezzo: ${price:.2f}<br>" +
                    f"Installazioni: {int(inst):,}<br>" +
                    f"Dimensione: {size:.1f}MB<br>" +
                    f"Giorni dall'ultimo aggiornamento: {days:.0f}"
                    for app, cat, rating, days, price, inst, size in zip(
                        df['App'],
                        df['Category'],
                        df['Rating'],
                        df['Days_Since_Update'],
                        df['Price_Clean'],
                        df['Installs_Clean'],
                        df['Size_MB']
                    )
                ]

                # Plot scatter points
                fig.add_trace(
                    go.Scatter(
                        x=jittered_x,
                        y=df[y],
                        mode='markers',
                        marker=dict(
                            size=3,
                            opacity=0.5,
                            color=df['Rating'],
                            colorscale=colorscale,
                            showscale=False
                        ),
                        hovertemplate="%{text}<extra></extra>",
                        text=hover_text,
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # Plot moving average
                fig.add_trace(
                    go.Scatter(
                        x=rating_range,
                        y=days_mean,
                        mode='lines',
                        line=dict(color='black', width=2),
                        showlegend=False,
                        hoverinfo='skip'  # Nascondi hover per la linea media
                    ),
                    row=row, col=col
                )

                fig.update_xaxes(
                    title=x.replace('_', ' '),
                    range=[1, 5],
                    row=row,
                    col=col,
                    gridcolor='lightgrey',
                    showgrid=True
                )

                fig.update_yaxes(
                    title=y.replace('_', ' '),
                    range=[0, 1000],
                    row=row,
                    col=col,
                    gridcolor='lightgrey',
                    showgrid=True
                )

            else:
                fig.add_trace(
                    go.Scatter(
                        x=df[x],
                        y=df[y],
                        mode='markers',
                        marker=dict(
                            size=3,
                            opacity=0.5,
                            color=df['Rating'],
                            colorscale=colorscale,
                            showscale=True if idx == 0 else False
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )

                fig.update_xaxes(
                    title=x.replace('_', ' '),
                    row=row,
                    col=col,
                    gridcolor='lightgrey',
                    showgrid=True
                )

                fig.update_yaxes(
                    title=y.replace('_', ' '),
                    row=row,
                    col=col,
                    gridcolor='lightgrey',
                    showgrid=True
                )

        fig.update_layout(
            height=800,
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(t=50, l=50, r=50, b=50)
        )

        return fig
    
    st.plotly_chart(create_scatter_matrix(filtered_df), use_container_width=True)

with tabs[2]:
    st.header("Analisi temporale")
    
    @st.cache_data
    def create_temporal_evolution(df):
        """Crea il grafico dell'evoluzione temporale delle metriche chiave"""
        # Prima prepariamo i dati
        df = df.copy()
        
        # Assicuriamoci che Last Updated sia datetime
        df['Last Updated'] = pd.to_datetime(df['Last Updated'])
        
        # Creiamo la colonna Update_Year se non esiste
        if 'Update_Year' not in df.columns:
            df['Update_Year'] = df['Last Updated'].dt.year
        
        # Calcolo metriche temporali aggregate per anno
        time_metrics = df.groupby('Update_Year').agg({
            'Rating': 'mean',
            'Size_MB': 'mean',
            'Installs_Clean': 'mean',
            'App': 'count'
        }).round(2)
        
        # Prezzi medi solo per app a pagamento
        avg_price = df[df['Price_Clean'] > 0].groupby('Update_Year')['Price_Clean'].mean()
        time_metrics['Price_Clean'] = avg_price
        
        # Creazione grafico con subplot
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.12,
            subplot_titles=(
                'Metriche di prodotto (Rating, Prezzo, Dimensione)',
                'Metriche di mercato (Installazioni e numero di app)'
            )
        )
        
        # Configurazione tracce subplot 1 - Metriche di prodotto
        traces_subplot1 = [
            ('Rating', 'Rating medio', '#2ECC40'),
            ('Price_Clean', 'Prezzo medio ($)', '#FF4136'),
            ('Size_MB', 'Dimensione media (MB)', '#0074D9')
        ]
        
        # Aggiunta tracce subplot 1 - Metriche di prodotto
        for col, name, color in traces_subplot1:
            data = time_metrics[col].fillna(0)
            
            fig.add_trace(
                go.Scatter(
                    x=time_metrics.index,
                    y=data,
                    name=name,
                    line=dict(color=color, width=2)
                ),
                row=1,
                col=1
            )
        
        # Aggiunta barre numero app subplot 2 - Metriche di mercato
        fig.add_trace(
            go.Bar(
                x=time_metrics.index,
                y=time_metrics['App'],
                name='Numero di app',
                marker_color='#AAAAAA',
                opacity=0.3,
                width=0.5
            ),
            row=2,
            col=1
        )
        
        # Aggiunta linea installazioni subplot 2 - Metriche di mercato
        fig.add_trace(
            go.Scatter(
                x=time_metrics.index,
                y=time_metrics['Installs_Clean'],
                name='Installazioni medie',
                line=dict(color='#FF851B', width=2)
            ),
            row=2,
            col=1
        )
        
        # Ottimizzazione layout
        fig.update_layout(
            height=900,
            showlegend=True,
            plot_bgcolor='white',
            title={
                'text': 'Evoluzione temporale delle metriche chiave',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.05,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            margin=dict(t=120, b=50, l=50, r=50),
            hovermode='x unified'
        )
        
        # Ottimizzazione assi
        for row in [1, 2]:
            fig.update_xaxes(
                title='Anno',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                row=row,
                col=1
            )
        
        fig.update_yaxes(
            title='Valore',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=1,
            col=1
        )
        
        fig.update_yaxes(
            title='Numero',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            type='log',
            row=2,
            col=1
        )
        
        return fig

    st.plotly_chart(create_temporal_evolution(filtered_df), use_container_width=True)

with tabs[3]:
    st.header("Analisi tecnica")
    
    # Prima il grafico delle versioni Android
    st.subheader("Distribuzione versioni Android")
    
    @st.cache_data
    def create_android_distribution(df):
        # Ordina le categorie come nell'immagine
        categories_order = [
            'LIBRARIES_AND_DEMO', 'BOOKS_AND_REFERENCE', 'COMMUNICATION', 'GAME', 'FAMILY',
            'PERSONALIZATION', 'TOOLS', 'ART_AND_DESIGN', 'LIFESTYLE', 'PHOTOGRAPHY',
            'WEATHER', 'COMICS', 'PRODUCTIVITY', 'BEAUTY', 'EDUCATION', 'BUSINESS',
            'MAPS_AND_NAVIGATION', 'SOCIAL', 'VIDEO_PLAYERS', 'HOUSE_AND_HOME',
            'PARENTING', 'MEDICAL', 'SPORTS', 'EVENTS', 'SHOPPING', 'AUTO_AND_VEHICLES',
            'HEALTH_AND_FITNESS', 'DATING', 'NEWS_AND_MAGAZINES', 'FINANCE',
            'TRAVEL_AND_LOCAL', 'FOOD_AND_DRINK', 'ENTERTAINMENT'
        ]
        
        # Calcola distribuzione
        android_dist = pd.crosstab(
            df['Category'], 
            df['Android_Ver_Clean'],
            normalize='index'
        ) * 100
        
        # Filtra le categorie presenti e riordina
        available_categories = [cat for cat in categories_order if cat in android_dist.index]
        android_dist = android_dist.loc[available_categories]
        
        fig = go.Figure()
        
        # Colori esatti come nell'immagine
        version_colors = {
            1.0: '#FF0000', 1.5: '#FF0000', 1.6: '#FF0000',  # Rosso
            2.0: '#FF0000', 2.1: '#FF0000', 2.2: '#FF0000', 2.3: '#FF0000',
            3.0: '#CD853F', 3.1: '#CD853F', 3.2: '#CD853F',  # Marrone
            4.0: '#8B8B00', 4.1: '#8B8B00', 4.2: '#8B8B00', 4.3: '#8B8B00', 4.4: '#8B8B00',  # Verde oliva
            5.0: '#00FF00', 5.1: '#00FF00',  # Verde chiaro
            6.0: '#00FF00', 7.0: '#00FF00', 7.1: '#00FF00', 8.0: '#00FF00'  # Verde
        }
        
        # Aggiungi le barre in ordine crescente di versione
        for version in sorted(df['Android_Ver_Clean'].unique()):
            if pd.notna(version):
                fig.add_trace(
                    go.Bar(
                        name=f'Android {version}',
                        x=android_dist[version],
                        y=android_dist.index,
                        orientation='h',
                        marker_color=version_colors.get(version, '#808080'),
                        text=android_dist[version].round(1).astype(str) + '%',
                        textposition='auto',
                        textfont=dict(size=10)
                    )
                )
        
        fig.update_layout(
            barmode='stack',
            height=800,
            margin=dict(t=20, l=200, r=100, b=50),
            plot_bgcolor='white',
            yaxis_title='Categoria',
            xaxis_title='Percentuale di app (%)',
            showlegend=True,
            legend=dict(
                title='Versione Android',
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='right',
                x=1.1
            ),
            xaxis=dict(
                range=[0, 100],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                autorange="reversed",  # Mantiene l'ordine delle categorie
                gridcolor='lightgrey'
            )
        )
        
        return fig
    
    st.plotly_chart(create_android_distribution(filtered_df), use_container_width=True)
    
    # Poi il grafico delle dimensioni
    st.subheader("Distribuzione dimensioni app")
    
    @st.cache_data
    def create_size_distribution(df):
        fig = go.Figure()
        
        # Usa lo stesso ordine delle categorie del grafico precedente
        categories_order = [
            'LIBRARIES_AND_DEMO', 'BOOKS_AND_REFERENCE', 'COMMUNICATION', 'GAME', 'FAMILY',
            'PERSONALIZATION', 'TOOLS', 'ART_AND_DESIGN', 'LIFESTYLE', 'PHOTOGRAPHY',
            'WEATHER', 'COMICS', 'PRODUCTIVITY', 'BEAUTY', 'EDUCATION', 'BUSINESS',
            'MAPS_AND_NAVIGATION', 'SOCIAL', 'VIDEO_PLAYERS', 'HOUSE_AND_HOME',
            'PARENTING', 'MEDICAL', 'SPORTS', 'EVENTS', 'SHOPPING', 'AUTO_AND_VEHICLES',
            'HEALTH_AND_FITNESS', 'DATING', 'NEWS_AND_MAGAZINES', 'FINANCE',
            'TRAVEL_AND_LOCAL', 'FOOD_AND_DRINK', 'ENTERTAINMENT'
        ]
        
        for category in categories_order:
            category_data = df[df['Category'] == category]
            category_sizes = category_data['Size_MB'].dropna()
            
            fig.add_trace(
                go.Box(
                    y=category_sizes,
                    name=category,
                    boxpoints='outliers',
                    jitter=0.3,
                    hovertemplate=(
                        "App: %{customdata[0]}<br>" +
                        "Dimensione: %{y:.1f} MB<br>" +
                        "Rating: %{customdata[1]:.1f}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=category_data[['App', 'Rating']].values
                )
            )
        
        fig.update_layout(
            height=800,
            margin=dict(t=20, l=50, r=50, b=100),
            xaxis_tickangle=-45,
            yaxis_title='Dimensione (MB)',
            plot_bgcolor='white',
            xaxis_title='Categoria',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        )
        
        return fig
    
    st.plotly_chart(create_size_distribution(filtered_df), use_container_width=True)