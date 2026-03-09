import matplotlib.pyplot as plt 

def create_probability_plot(results, thresholds, title):                
    colors = ['red', 'orange', 'green']  # Colors for the thresholds
    
    plt.figure(figsize=(9, 6))
    
    # Plot correct first
    plt.scatter(results['proba'], results.index, c=results['proba'], cmap='magma', alpha=0.5)    

    # Plot thresholds
    for threshold, color in zip(thresholds, colors):
        plt.axvline(x=threshold, color=color, linestyle='--', label=f'Threshold {threshold}')

    
    plt.xlabel("Probability of being EXTENDED (M)")
    plt.xlim(0,1)
    plt.ylabel("Sample index (display order)")
    plt.title(f"Prediction Confidence: {plt.title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/'+title+'.png')
    plt.show() 