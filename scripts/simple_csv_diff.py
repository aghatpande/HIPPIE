#!/usr/bin/env python3
"""
Simple CSV Diff Script
"""

import pandas as pd
import sys

def simple_csv_diff(file1, file2):
    """Simple CSV comparison function."""
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        print(f"Comparing {file1} vs {file2}")
        print(f"File 1: {df1.shape[0]} rows, {df1.shape[1]} columns")
        print(f"File 2: {df2.shape[0]} rows, {df2.shape[1]} columns")
        
        if df1.equals(df2):
            print("✓ Files are identical")
            return True
        
        print("✗ Files differ:")
        
        # Shape differences
        if df1.shape != df2.shape:
            print(f"  Shape: {df1.shape} vs {df2.shape}")
        
        # Column differences
        cols1, cols2 = set(df1.columns), set(df2.columns)
        if cols1 != cols2:
            print(f"  Columns only in file 1: {list(cols1 - cols2)}")
            print(f"  Columns only in file 2: {list(cols2 - cols1)}")
        
        # Value differences (for common columns)
        common_cols = list(cols1 & cols2)
        if common_cols:
            df1_common = df1[common_cols]
            df2_common = df2[common_cols]
            
            # Align shapes for comparison
            min_rows = min(len(df1_common), len(df2_common))
            df1_aligned = df1_common.iloc[:min_rows]
            df2_aligned = df2_common.iloc[:min_rows]
            
            differences = (df1_aligned != df2_aligned).sum().sum()
            if differences > 0:
                print(f"  Value differences: {differences} cells")
        
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_diff.py <file1.csv> <file2.csv>")
        sys.exit(1)
    
    simple_csv_diff(sys.argv[1], sys.argv[2])