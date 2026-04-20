import matplotlib.pyplot as plt

def plot_target_distribution(df):
    plt.figure(figsize=(6, 4))
    df['Risk'].value_counts().plot(kind='bar')
    plt.title('Distribution of Risk')
    plt.xlabel('Risk')
    plt.ylabel('Count')
    plt.show()
    plt.savefig("./reports/figures/target_distribution.png")
    