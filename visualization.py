"""
Visualization Module for Online Sales Data Analysis Application

This module provides functions for creating various visualizations, including:
- Time series plots
- Bar charts and histograms
- Scatter plots and correlation heatmaps
- Pie charts and donut charts
- Geographic visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set default style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def create_time_series_plot(df, date_column, value_column, title=None, color='#1f77b4'):
    """
    Create an interactive time series plot using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    date_column : str
        Name of the date column
    value_column : str
        Name of the value column to plot
    title : str, optional
        Plot title
    color : str, default='#1f77b4'
        Line color
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if df is None or df.empty or date_column not in df.columns or value_column not in df.columns:
        return go.Figure()
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        try:
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            return go.Figure()
    
    # Sort by date
    df_sorted = df.sort_values(by=date_column)
    
    # Create the figure
    fig = px.line(
        df_sorted, 
        x=date_column, 
        y=value_column,
        markers=True,
        title=title or f'{value_column} over Time',
        labels={date_column: 'Date', value_column: value_column.replace('_', ' ').title()},
        line_shape='linear'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=value_column.replace('_', ' ').title(),
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def create_bar_chart(df, x_column, y_column, color_column=None, title=None, orientation='v'):
    """
    Create an interactive bar chart using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    x_column : str
        Name of the column for x-axis
    y_column : str
        Name of the column for y-axis
    color_column : str, optional
        Name of the column for color encoding
    title : str, optional
        Plot title
    orientation : str, default='v'
        Bar orientation ('v' for vertical, 'h' for horizontal)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if df is None or df.empty or x_column not in df.columns or y_column not in df.columns:
        return go.Figure()
    
    if color_column and color_column not in df.columns:
        color_column = None
    
    # Create the figure
    fig = px.bar(
        df, 
        x=x_column if orientation == 'v' else y_column,
        y=y_column if orientation == 'v' else x_column,
        color=color_column,
        title=title or f'{y_column} by {x_column}',
        labels={
            x_column: x_column.replace('_', ' ').title(),
            y_column: y_column.replace('_', ' ').title()
        },
        orientation=orientation
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column.replace('_', ' ').title() if orientation == 'v' else y_column.replace('_', ' ').title(),
        yaxis_title=y_column.replace('_', ' ').title() if orientation == 'v' else x_column.replace('_', ' ').title(),
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

def create_histogram(df, column, bins=20, title=None, color='#1f77b4'):
    """
    Create an interactive histogram using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    column : str
        Name of the column to plot
    bins : int, default=20
        Number of bins
    title : str, optional
        Plot title
    color : str, default='#1f77b4'
        Bar color
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if df is None or df.empty or column not in df.columns:
        return go.Figure()
    
    # Create the figure
    fig = px.histogram(
        df, 
        x=column,
        nbins=bins,
        title=title or f'Distribution of {column}',
        labels={column: column.replace('_', ' ').title()},
        color_discrete_sequence=[color]
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title='Count',
        template='plotly_white',
        bargap=0.1
    )
    
    # Add a box plot on the second y-axis
    fig.add_trace(
        go.Box(
            x=df[column],
            name='Box Plot',
            marker_color=color,
            boxpoints='outliers',
            jitter=0,
            fillcolor='rgba(255,255,255,0)',
            line={'color': 'rgba(0,0,0,0)'},
            showlegend=False,
            y0=0
        )
    )
    
    return fig

def create_scatter_plot(df, x_column, y_column, color_column=None, size_column=None, title=None):
    """
    Create an interactive scatter plot using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    x_column : str
        Name of the column for x-axis
    y_column : str
        Name of the column for y-axis
    color_column : str, optional
        Name of the column for color encoding
    size_column : str, optional
        Name of the column for size encoding
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if df is None or df.empty or x_column not in df.columns or y_column not in df.columns:
        return go.Figure()
    
    if color_column and color_column not in df.columns:
        color_column = None
    
    if size_column and size_column not in df.columns:
        size_column = None
    
    # Create the figure
    fig = px.scatter(
        df, 
        x=x_column,
        y=y_column,
        color=color_column,
        size=size_column,
        title=title or f'{y_column} vs {x_column}',
        labels={
            x_column: x_column.replace('_', ' ').title(),
            y_column: y_column.replace('_', ' ').title()
        },
        hover_data=df.columns
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column.replace('_', ' ').title(),
        yaxis_title=y_column.replace('_', ' ').title(),
        template='plotly_white',
        hovermode='closest'
    )
    
    # Add trend line
    if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='x', yref='y',
                    x0=df[x_column].min(),
                    y0=np.polyval(np.polyfit(df[x_column], df[y_column], 1), df[x_column].min()),
                    x1=df[x_column].max(),
                    y1=np.polyval(np.polyfit(df[x_column], df[y_column], 1), df[x_column].max()),
                    line=dict(color='red', width=2, dash='dash')
                )
            ]
        )
    
    return fig

def create_correlation_heatmap(df, columns=None, title=None, colorscale='RdBu_r'):
    """
    Create an interactive correlation heatmap using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    columns : list, optional
        List of column names to include. If None, all numeric columns are used.
    title : str, optional
        Plot title
    colorscale : str, default='RdBu_r'
        Colorscale for the heatmap
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if df is None or df.empty:
        return go.Figure()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    else:
        # Filter to include only existing numeric columns
        columns = [col for col in columns if col in df.columns and 
                  pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return go.Figure()
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create the figure
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale=colorscale,
        title=title or 'Correlation Matrix',
        zmin=-1, zmax=1
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
        template='plotly_white'
    )
    
    # Add correlation values as text
    for i, row in enumerate(corr_matrix.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=corr_matrix.columns[j],
                y=corr_matrix.columns[i],
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color='white' if abs(val) > 0.5 else 'black')
            )
    
    return fig

def create_pie_chart(df, names_column, values_column, title=None, hole=0.4):
    """
    Create an interactive pie or donut chart using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    names_column : str
        Name of the column for slice names
    values_column : str
        Name of the column for slice values
    title : str, optional
        Plot title
    hole : float, default=0.4
        Size of the hole (0 for pie chart, >0 for donut chart)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if df is None or df.empty or names_column not in df.columns or values_column not in df.columns:
        return go.Figure()
    
    # Create the figure
    fig = px.pie(
        df, 
        names=names_column,
        values=values_column,
        title=title or f'{values_column} by {names_column}',
        hole=hole
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        legend_title=names_column.replace('_', ' ').title()
    )
    
    # Update traces
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        marker=dict(line=dict(color='white', width=2))
    )
    
    return fig

def create_box_plot(df, x_column, y_column, color_column=None, title=None):
    """
    Create an interactive box plot using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    x_column : str
        Name of the column for x-axis (categories)
    y_column : str
        Name of the column for y-axis (values)
    color_column : str, optional
        Name of the column for color encoding
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if df is None or df.empty or x_column not in df.columns or y_column not in df.columns:
        return go.Figure()
    
    if color_column and color_column not in df.columns:
        color_column = None
    
    # Create the figure
    fig = px.box(
        df, 
        x=x_column,
        y=y_column,
        color=color_column,
        title=title or f'Distribution of {y_column} by {x_column}',
        labels={
            x_column: x_column.replace('_', ' ').title(),
            y_column: y_column.replace('_', ' ').title()
        },
        points='outliers'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column.replace('_', ' ').title(),
        yaxis_title=y_column.replace('_', ' ').title(),
        template='plotly_white',
        boxmode='group'
    )
    
    return fig

def create_multi_chart_dashboard(df, chart_specs, title=None, rows=None, cols=None):
    """
    Create a dashboard with multiple charts using Plotly subplots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    chart_specs : list of dict
        List of chart specifications, each containing:
        - 'type': Chart type ('bar', 'line', 'scatter', 'pie', 'histogram', 'box')
        - 'x': x-axis column (if applicable)
        - 'y': y-axis column (if applicable)
        - 'color': color column (optional)
        - 'title': chart title (optional)
        - 'row': row position (1-based, optional)
        - 'col': column position (1-based, optional)
    title : str, optional
        Dashboard title
    rows : int, optional
        Number of rows in the dashboard
    cols : int, optional
        Number of columns in the dashboard
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if df is None or df.empty or not chart_specs:
        return go.Figure()
    
    # Determine grid dimensions if not specified
    if rows is None and cols is None:
        n_charts = len(chart_specs)
        cols = min(3, n_charts)
        rows = (n_charts + cols - 1) // cols
    elif rows is None:
        n_charts = len(chart_specs)
        rows = (n_charts + cols - 1) // cols
    elif cols is None:
        n_charts = len(chart_specs)
        cols = (n_charts + rows - 1) // rows
    
    # Create subplot figure
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[spec.get('title', f"Chart {i+1}") for i, spec in enumerate(chart_specs)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Add charts to the dashboard
    for i, spec in enumerate(chart_specs):
        chart_type = spec.get('type', 'bar')
        row = spec.get('row', i // cols + 1)
        col = spec.get('col', i % cols + 1)
        
        try:
            if chart_type == 'bar':
                x = spec.get('x')
                y = spec.get('y')
                color = spec.get('color')
                orientation = spec.get('orientation', 'v')
                
                if x and y and x in df.columns and y in df.columns:
                    if orientation == 'v':
                        fig.add_trace(
                            go.Bar(
                                x=df[x],
                                y=df[y],
                                marker_color=df[color] if color and color in df.columns else None,
                                name=spec.get('title', f"{y} by {x}")
                            ),
                            row=row, col=col
                        )
                    else:
                        fig.add_trace(
                            go.Bar(
                                x=df[y],
                                y=df[x],
                                marker_color=df[color] if color and color in df.columns else None,
                                name=spec.get('title', f"{y} by {x}"),
                                orientation='h'
                            ),
                            row=row, col=col
                        )
                
            elif chart_type == 'line':
                x = spec.get('x')
                y = spec.get('y')
                
                if x and y and x in df.columns and y in df.columns:
                    df_sorted = df.sort_values(by=x)
                    fig.add_trace(
                        go.Scatter(
                            x=df_sorted[x],
                            y=df_sorted[y],
                            mode='lines+markers',
                            name=spec.get('title', f"{y} over {x}")
                        ),
                        row=row, col=col
                    )
                
            elif chart_type == 'scatter':
                x = spec.get('x')
                y = spec.get('y')
                color = spec.get('color')
                size = spec.get('size')
                
                if x and y and x in df.columns and y in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df[x],
                            y=df[y],
                            mode='markers',
                            marker=dict(
                                color=df[color] if color and color in df.columns else None,
                                size=df[size] if size and size in df.columns else None
                            ),
                            name=spec.get('title', f"{y} vs {x}")
                        ),
                        row=row, col=col
                    )
                
            elif chart_type == 'pie':
                names = spec.get('names')
                values = spec.get('values')
                hole = spec.get('hole', 0.4)
                
                if names and values and names in df.columns and values in df.columns:
                    fig.add_trace(
                        go.Pie(
                            labels=df[names],
                            values=df[values],
                            hole=hole,
                            name=spec.get('title', f"{values} by {names}")
                        ),
                        row=row, col=col
                    )
                
            elif chart_type == 'histogram':
                x = spec.get('x')
                bins = spec.get('bins', 20)
                
                if x and x in df.columns:
                    fig.add_trace(
                        go.Histogram(
                            x=df[x],
                            nbinsx=bins,
                            name=spec.get('title', f"Distribution of {x}")
                        ),
                        row=row, col=col
                    )
                
            elif chart_type == 'box':
                x = spec.get('x')
                y = spec.get('y')
                color = spec.get('color')
                
                if y and y in df.columns:
                    if x and x in df.columns:
                        fig.add_trace(
                            go.Box(
                                x=df[x],
                                y=df[y],
                                marker_color=df[color] if color and color in df.columns else None,
                                name=spec.get('title', f"Distribution of {y}")
                            ),
                            row=row, col=col
                        )
                    else:
                        fig.add_trace(
                            go.Box(
                                y=df[y],
                                name=spec.get('title', f"Distribution of {y}")
                            ),
                            row=row, col=col
                        )
        
        except Exception as e:
            print(f"Error creating chart {i+1}: {e}")
    
    # Update layout
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=300 * rows,
        width=400 * cols,
        showlegend=False
    )
    
    return fig
