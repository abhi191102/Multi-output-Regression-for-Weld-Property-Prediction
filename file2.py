import matplotlib.pyplot as plt

for i, col in enumerate(['BW', 'BH', 'UTS']):
    plt.figure(figsize=(5, 4))
    plt.scatter(y_test[col], y_pred[:, i], alpha=0.7, color='skyblue')
    plt.plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{col} - Actual vs Predicted (Random Forest)')
    plt.grid(True)
    plt.show()
