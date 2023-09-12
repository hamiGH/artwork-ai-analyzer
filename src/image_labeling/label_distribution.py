from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import ast


def plot_label_distribution(dataset_path: str):
    art_dataset = pd.read_csv(dataset_path)
    art_dataset['category_labels'] = art_dataset['category_labels'].apply(ast.literal_eval)

    label_counts = Counter(label for labels in art_dataset['category_labels'] for label in labels)

    # Sort labels and counts in descending order by frequency
    sorted_labels_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_labels_counts)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Label Distribution')
    # plt.grid(axis='y', linestyle='--', alpha=0.6, which='both', color='gray', linewidth=0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Calculate the total number of labels
    total_labels = sum(label_counts.values())

    # Calculate the percentages for each label
    label_percentages = {label: count / total_labels * 100 for label, count in label_counts.items()}

    # Sort labels and percentages in descending order by percentage
    sorted_label_percentages = sorted(label_percentages.items(), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_percentages = zip(*sorted_label_percentages)

    # Create a bar plot to visualize the label percentages with 50 vertical grid lines
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_labels, sorted_percentages)
    plt.xlabel('Labels')
    plt.ylabel('Percentage')
    plt.title('Label Distribution as Percentage (Descending Order)')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.gca().xaxis.grid(True, which='major')
    # plt.gca().xaxis.grid(True, which='minor', linestyle='--')
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.grid(True, which='major')
    plt.gca().yaxis.grid(True, which='minor', linestyle='--')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
