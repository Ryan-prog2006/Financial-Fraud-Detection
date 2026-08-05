import pandas as pd
import numpy as np
import os

def main():
    os.makedirs('data/raw', exist_ok=True)
    n = 1000
    data = {f'V{i}': np.random.randn(n) for i in range(1, 29)}
    data['Time'] = np.arange(n) * 10.0
    data['Amount'] = np.random.uniform(1, 500, n)
    data['Class'] = np.where(np.arange(n) < 10, 1, 0)
    df = pd.DataFrame(data)
    df.to_csv('data/raw/creditcard.csv', index=False)
    print("Synthetic creditcard.csv generated successfully.")

if __name__ == '__main__':
    main()
