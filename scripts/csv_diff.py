#!/usr/bin/env python3
"""
CSV Diff Generator
Compares two CSV files and generates a detailed diff report.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
import difflib
from typing import Tuple, List, Dict, Any

class CSVDiff:
    def __init__(self, file1: str, file2: str, key_column: str = None):
        """
        Initialize CSV diff generator.
        
        Args:
            file1: Path to first CSV file
            file2: Path to second CSV file
            key_column: Column to use as unique identifier for row-based comparison
        """
        self.file1 = file1
        self.file2 = file2
        self.key_column = key_column
        self.df1 = None
        self.df2 = None
        
    def load_files(self) -> bool:
        """Load CSV files into DataFrames."""
        try:
            self.df1 = pd.read_csv(self.file1)
            self.df2 = pd.read_csv(self.file2)
            print(f"✓ Loaded {self.file1} ({len(self.df1)} rows, {len(self.df1.columns)} columns)")
            print(f"✓ Loaded {self.file2} ({len(self.df2)} rows, {len(self.df2.columns)} columns)")
            return True
        except Exception as e:
            print(f"✗ Error loading files: {e}")
            return False
    
    def compare_structure(self) -> Dict[str, Any]:
        """Compare the structure of both CSV files."""
        structure_diff = {
            'shape_diff': {
                'file1': self.df1.shape,
                'file2': self.df2.shape,
                'rows_diff': self.df2.shape[0] - self.df1.shape[0],
                'cols_diff': self.df2.shape[1] - self.df1.shape[1]
            },
            'columns_diff': {
                'file1_only': list(set(self.df1.columns) - set(self.df2.columns)),
                'file2_only': list(set(self.df2.columns) - set(self.df1.columns)),
                'common': list(set(self.df1.columns) & set(self.df2.columns)),
                'order_changed': list(self.df1.columns) != list(self.df2.columns)
            }
        }
        return structure_diff
    
    def compare_data_types(self) -> Dict[str, Any]:
        """Compare data types of common columns."""
        common_cols = set(self.df1.columns) & set(self.df2.columns)
        dtype_diff = {}
        
        for col in common_cols:
            type1 = str(self.df1[col].dtype)
            type2 = str(self.df2[col].dtype)
            if type1 != type2:
                dtype_diff[col] = {'file1': type1, 'file2': type2}
        
        return dtype_diff
    
    def compare_values(self) -> Dict[str, Any]:
        """Compare values in the CSV files."""
        if self.key_column and self.key_column in self.df1.columns and self.key_column in self.df2.columns:
            return self._compare_by_key()
        else:
            return self._compare_by_position()
    
    def _compare_by_key(self) -> Dict[str, Any]:
        """Compare rows using a key column."""
        df1_keyed = self.df1.set_index(self.key_column)
        df2_keyed = self.df2.set_index(self.key_column)
        
        keys1 = set(df1_keyed.index)
        keys2 = set(df2_keyed.index)
        
        comparison = {
            'keys_only_in_file1': list(keys1 - keys2),
            'keys_only_in_file2': list(keys2 - keys1),
            'common_keys': list(keys1 & keys2),
            'modified_rows': []
        }
        
        # Compare common rows
        for key in comparison['common_keys']:
            row1 = df1_keyed.loc[key]
            row2 = df2_keyed.loc[key]
            
            differences = {}
            for col in row1.index:
                if col in row2.index:
                    if pd.isna(row1[col]) and pd.isna(row2[col]):
                        continue
                    elif row1[col] != row2[col]:
                        differences[col] = {
                            'file1': row1[col],
                            'file2': row2[col]
                        }
            
            if differences:
                comparison['modified_rows'].append({
                    'key': key,
                    'differences': differences
                })
        
        return comparison
    
    def _compare_by_position(self) -> Dict[str, Any]:
        """Compare DataFrames by position (row and column index)."""
        common_cols = list(set(self.df1.columns) & set(self.df2.columns))
        
        # Align DataFrames to have same columns for comparison
        df1_aligned = self.df1[common_cols] if common_cols else pd.DataFrame()
        df2_aligned = self.df2[common_cols] if common_cols else pd.DataFrame()
        
        comparison = {
            'common_columns': common_cols,
            'cell_differences': []
        }
        
        min_rows = min(len(df1_aligned), len(df2_aligned))
        
        for row_idx in range(min_rows):
            for col in common_cols:
                val1 = df1_aligned.iloc[row_idx][col]
                val2 = df2_aligned.iloc[row_idx][col]
                
                if pd.isna(val1) and pd.isna(val2):
                    continue
                elif val1 != val2:
                    comparison['cell_differences'].append({
                        'row': row_idx,
                        'column': col,
                        'file1_value': val1,
                        'file2_value': val2
                    })
        
        return comparison
    
    def generate_text_diff(self) -> str:
        """Generate a text-based diff similar to Unix diff."""
        try:
            with open(self.file1, 'r') as f1, open(self.file2, 'r') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
            
            diff = list(difflib.unified_diff(
                lines1, 
                lines2, 
                fromfile=self.file1, 
                tofile=self.file2,
                lineterm=''
            ))
            
            return '\n'.join(diff)
        except Exception as e:
            return f"Error generating text diff: {e}"
    
    def generate_report(self) -> str:
        """Generate a comprehensive diff report."""
        if not self.load_files():
            return "Failed to load files"
        
        report = []
        report.append("=" * 80)
        report.append("CSV FILES COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"File 1: {self.file1}")
        report.append(f"File 2: {self.file2}")
        report.append(f"Key Column: {self.key_column if self.key_column else 'None (positional comparison)'}")
        report.append("")
        
        # Structure comparison
        structure_diff = self.compare_structure()
        report.append("STRUCTURE COMPARISON")
        report.append("-" * 40)
        report.append(f"File 1 shape: {structure_diff['shape_diff']['file1']}")
        report.append(f"File 2 shape: {structure_diff['shape_diff']['file2']}")
        report.append(f"Rows difference: {structure_diff['shape_diff']['rows_diff']}")
        report.append(f"Columns difference: {structure_diff['shape_diff']['cols_diff']}")
        report.append("")
        
        # Column differences
        col_diff = structure_diff['columns_diff']
        if col_diff['file1_only']:
            report.append(f"Columns only in file 1: {col_diff['file1_only']}")
        if col_diff['file2_only']:
            report.append(f"Columns only in file 2: {col_diff['file2_only']}")
        if col_diff['order_changed']:
            report.append("Column order has changed")
        report.append("")
        
        # Data type comparison
        dtype_diff = self.compare_data_types()
        if dtype_diff:
            report.append("DATA TYPE DIFFERENCES")
            report.append("-" * 40)
            for col, types in dtype_diff.items():
                report.append(f"{col}: {types['file1']} → {types['file2']}")
            report.append("")
        
        # Value comparison
        value_diff = self.compare_values()
        report.append("VALUE COMPARISON")
        report.append("-" * 40)
        
        if self.key_column:
            if value_diff.get('keys_only_in_file1'):
                report.append(f"Keys only in file 1: {len(value_diff['keys_only_in_file1'])}")
                for key in value_diff['keys_only_in_file1'][:10]:  # Show first 10
                    report.append(f"  - {key}")
                if len(value_diff['keys_only_in_file1']) > 10:
                    report.append(f"  ... and {len(value_diff['keys_only_in_file1']) - 10} more")
            
            if value_diff.get('keys_only_in_file2'):
                report.append(f"Keys only in file 2: {len(value_diff['keys_only_in_file2'])}")
                for key in value_diff['keys_only_in_file2'][:10]:  # Show first 10
                    report.append(f"  + {key}")
                if len(value_diff['keys_only_in_file2']) > 10:
                    report.append(f"  ... and {len(value_diff['keys_only_in_file2']) - 10} more")
            
            if value_diff.get('modified_rows'):
                report.append(f"Modified rows: {len(value_diff['modified_rows'])}")
                for mod_row in value_diff['modified_rows'][:5]:  # Show first 5
                    report.append(f"  Key: {mod_row['key']}")
                    for col, change in mod_row['differences'].items():
                        report.append(f"    {col}: {change['file1']} → {change['file2']}")
                if len(value_diff['modified_rows']) > 5:
                    report.append(f"  ... and {len(value_diff['modified_rows']) - 5} more modified rows")
        else:
            if value_diff.get('cell_differences'):
                report.append(f"Cell differences: {len(value_diff['cell_differences'])}")
                for diff in value_diff['cell_differences'][:10]:  # Show first 10
                    report.append(f"  Row {diff['row']}, Col '{diff['column']}': {diff['file1_value']} → {diff['file2_value']}")
                if len(value_diff['cell_differences']) > 10:
                    report.append(f"  ... and {len(value_diff['cell_differences']) - 10} more differences")
        
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        if self.df1.equals(self.df2):
            report.append("✓ Files are identical")
        else:
            report.append("✗ Files differ")
            changes = []
            if structure_diff['shape_diff']['rows_diff'] != 0:
                changes.append(f"row count ({structure_diff['shape_diff']['rows_diff']:+d})")
            if structure_diff['shape_diff']['cols_diff'] != 0:
                changes.append(f"column count ({structure_diff['shape_diff']['cols_diff']:+d})")
            if dtype_diff:
                changes.append(f"data types ({len(dtype_diff)} columns)")
            if self.key_column:
                if value_diff.get('modified_rows'):
                    changes.append(f"modified rows ({len(value_diff['modified_rows'])})")
            else:
                if value_diff.get('cell_differences'):
                    changes.append(f"cell values ({len(value_diff['cell_differences'])})")
            
            if changes:
                report.append(f"Changes detected in: {', '.join(changes)}")
        
        return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='Generate diff between two CSV files')
    parser.add_argument('file1', help='First CSV file')
    parser.add_argument('file2', help='Second CSV file')
    parser.add_argument('-k', '--key', help='Column to use as unique identifier')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--text-diff', action='store_true', help='Also generate text-based diff')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.file1).exists():
        print(f"Error: File '{args.file1}' not found")
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"Error: File '{args.file2}' not found")
        sys.exit(1)
    
    # Generate diff
    csv_diff = CSVDiff(args.file1, args.file2, args.key)
    report = csv_diff.generate_report()
    
    if args.text_diff:
        report += "\n\n" + "=" * 80 + "\n"
        report += "TEXT-BASED DIFF\n"
        report += "=" * 80 + "\n"
        report += csv_diff.generate_text_diff()
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Diff report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()