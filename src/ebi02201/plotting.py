import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.offline as pyo

def plot_ner_counts(dataframe: pd.DataFrame, ner_column: str, sort_counts: bool = True, 
                    palette: str = "viridis", figsize: tuple = (10, 6), 
                    title: Optional[str] = None, xlabel: Optional[str] = None, 
                    ylabel: Optional[str] = None, rotation: int = 45) -> None:
    """
    Plots the count of NER categories from a given DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing the NER data.
    - ner_column (str): Name of the column in DataFrame that contains NER categories.
    - sort_counts (bool): If True, sorts the bars by counts.
    - palette (str): Color palette for the bars.
    - figsize (tuple): Size of the figure.
    - title (str): Title of the plot. If None, a default title is used.
    - xlabel (str): Label for the x-axis. If None, a default label is used.
    - ylabel (str): Label for the y-axis. If None, a default label is used.
    - rotation (int): Degree of rotation for x-axis labels.
    """

    # Count of NER categories
    ner_counts = dataframe[ner_column].value_counts()

    # Sorting if required
    if sort_counts:
        ner_counts = ner_counts.sort_values()

    # Setting the aesthetic style
    sns.set_style("whitegrid")

    # Create figure
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=ner_counts.index, y=ner_counts.values, palette=palette)

    # Setting title and labels
    plt.title(title or 'Count of NER Categories', fontsize=14)
    plt.xlabel(xlabel or 'NER Category', fontsize=12)
    plt.ylabel(ylabel or 'Counts', fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=rotation)

    # Adding count labels on top of bars
    for i, count in enumerate(ner_counts.values):
        ax.text(i, count, count, ha='center', va='bottom')

    # Show plot
    plt.tight_layout()
    plt.show()

def visualize_probabilities_and_labels(filename, probabilities, labels, idx, annotation_types, tokenizer, max_seq_length, final_df, num_tokens=64, colors=None, title_font_size=10):
    # Get the sentence, probabilities, and labels for the specified index
    sentence = tokenizer.tokenize(final_df.loc[idx, 'sentence'], add_special_tokens=True, max_length=max_seq_length)
    probs = probabilities[idx].detach().numpy()
    label_tensor = labels[idx]

    num_to_visualize = min(num_tokens, len(sentence))

    # Create a DataFrame for visualization
    df = pd.DataFrame({'Token': sentence[:num_to_visualize]})

    for i, label in enumerate(annotation_types):
        prob_col_name = f'Probability_{label}'
        label_col_name = f'Label_{label}'
        df[prob_col_name] = probs[:num_to_visualize, i]
        df[label_col_name] = label_tensor[:num_to_visualize, i].numpy()

    # Create subplots with shared x-axis
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    if colors is None:
        colors = ['blue', 'red', 'green', 'orange']

    # Add traces for probabilities and labels
    for i, label in enumerate(annotation_types):
        prob_col_name = f'Probability_{label}'
        label_col_name = f'Label_{label}'
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df[prob_col_name], mode='lines', name=f'Probability {label}', line=dict(color=colors[i], dash='dash')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df[label_col_name], mode='lines', name=f'Label {label}', line=dict(color=colors[i])),
            row=2, col=1
        )

    # Update axis labels and title
    fig.update_xaxes(title='Token Index', tickvals=df.index, ticktext=df['Token'], tickangle=30)
    fig.update_yaxes(title='Probability/Label', range=[0, 1])

    sentence = final_df.loc[idx, 'sentence']
    ner = final_df.loc[idx, 'ner']
    
    fig.update_layout(
        title_text=f'Sentence: {sentence}\nNER: {repr(ner)}',
        title_font_size=title_font_size,
    )
    
    # Show the plot
    pyo.plot(fig, filename=filename)
    fig.show()