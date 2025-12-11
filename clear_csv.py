#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Minimal RAM TSE CSV Reducer - Direct File Streaming

Uses raw file I/O with minimal pandas usage - literally never loads > 5MB into RAM

Perfect for 8GB systems where 92% RAM is unacceptable
"""

import csv
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_PATH = Path("/home/otdsp/more-lula-more-women-?/data")
CONSULTA_CAND_PATH = BASE_PATH / "consulta_cand_2022"

# All Brazilian states
UFS = [
    'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
    'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN',
    'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO'
]

# ==============================================================================
# COLUMN DEFINITIONS
# ==============================================================================

COLS_PRESIDENTIAL = [
    'NR_TURNO', 'NRTURNO',
    'CD_CARGO', 'CDCARGO',
    'SG_UF', 'SGUF',
    'CD_MUNICIPIO', 'CDMUNICIPIO',
    'NM_MUNICIPIO', 'NMMUNICIPIO',
    'NR_VOTAVEL', 'NRVOTAVEL',
    'QT_VOTOS', 'QTVOTOS'
]

COLS_CANDIDATE = [
    'CD_CARGO', 'CDCARGO',
    'DS_CARGO', 'DSCARGO',
    'NR_TURNO', 'NRTURNO',
    'DS_GENERO', 'DSGENERO',
    'SG_UF', 'SGUF',
    'NR_CANDIDATO', 'NRCANDIDATO'
]

COLS_DEPUTY = [
    'NR_TURNO', 'NRTURNO',
    'CD_CARGO', 'CDCARGO',
    'SG_UF', 'SGUF',
    'CD_MUNICIPIO', 'CDMUNICIPIO',
    'NM_MUNICIPIO', 'NMMUNICIPIO',
    'NR_VOTAVEL', 'NRVOTAVEL',
    'QT_VOTOS', 'QTVOTOS'
]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_file_size_mb(filepath):
    """Get file size in MB"""
    if not filepath.exists():
        return 0
    return filepath.stat().st_size / (1024 * 1024)

def format_time(seconds):
    """Format seconds as HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def find_columns_in_header(header_row, col_options):
    """Find which columns from options exist in header"""
    cols_to_keep = []
    col_indices = {}
    
    for idx, col_name in enumerate(header_row):
        if col_name in col_options:
            cols_to_keep.append(col_name)
            col_indices[col_name] = idx
    
    return cols_to_keep, col_indices

def process_file_streaming(input_path, output_path, col_options, file_type="state"):
    """
    Process CSV using raw file I/O - never loads entire file into memory.
    Only keeps 1 row in memory at a time.
    RAM usage: ~5-10MB constant, never varies based on file size!
    """
    
    if not input_path.exists():
        return None
    
    original_size = get_file_size_mb(input_path)
    
    try:
        print(f" [Streaming {file_type}...] ", end='', flush=True)
        
        total_rows = 0
        rows_written = 0
        
        with open(input_path, 'r', encoding='latin1') as infile:
            reader = csv.reader(infile, delimiter=';')
            
            # Read header
            try:
                header_row = next(reader)
            except StopIteration:
                print(f" ✗ (empty file)")
                return None
            
            # Find which columns to keep
            cols_to_keep, col_indices = find_columns_in_header(header_row, col_options)
            
            if not cols_to_keep:
                print(f" ✗ (no matching columns)")
                return None
            
            # Write output file
            with open(output_path, 'w', encoding='utf-8-sig', newline='') as outfile:
                writer = csv.writer(outfile, delimiter=';')
                
                # Write header
                writer.writerow(cols_to_keep)
                
                # Process rows one at a time
                for row in reader:
                    total_rows += 1
                    
                    # Extract only needed columns
                    selected_row = []
                    for col_name in cols_to_keep:
                        col_idx = col_indices[col_name]
                        
                        # Handle rows shorter than expected
                        if col_idx < len(row):
                            selected_row.append(row[col_idx])
                        else:
                            selected_row.append('')
                    
                    writer.writerow(selected_row)
                    rows_written += 1
        
        # Get output size
        new_size = get_file_size_mb(output_path)
        reduction = ((original_size - new_size) / original_size * 100) if original_size > 0 else 0
        
        return {
            'rows': rows_written,
            'original_size': original_size,
            'new_size': new_size,
            'reduction': reduction,
            'columns_kept': len(cols_to_keep)
        }
    
    except Exception as e:
        print(f" ✗ ({str(e)})")
        
        # Clean up bad output file
        try:
            if output_path.exists():
                output_path.unlink()
        except:
            pass
        
        return None

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def main():
    print("=" * 100)
    print(" " * 20 + "MINIMAL RAM TSE CSV REDUCER")
    print(" " * 15 + "Direct File Streaming - Constant ~5-10MB RAM Usage")
    print("=" * 100)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Strategy: Raw CSV file I/O (no pandas dataframe loading)")
    print(f"RAM Usage: ~5-10MB constant (independent of file size)")
    print("=" * 100)
    
    start_time = time.time()
    
    stats = {
        'presidential': None,
        'candidates': None,
        'states': {},
        'total_original': 0,
        'total_reduced': 0,
        'total_rows': 0,
        'failed_states': [],
        'successful_states': 0
    }
    
    # [1] Presidential data
    print("\n[1/3] PRESIDENTIAL DATA (votacao_secao_2022_BR.csv)")
    print("-" * 100)
    
    pres_file = BASE_PATH / "votacao_secao_2022_BR.csv"
    output_file = BASE_PATH / "votacao_secao_2022_BR_REDUCED.csv"
    print(f" Processing: {pres_file.name}")
    
    result = process_file_streaming(pres_file, output_file, COLS_PRESIDENTIAL, "presidential")
    
    if result:
        print(f"✓")
        print(f" Original: {result['original_size']:.1f}MB | Reduced: {result['new_size']:.1f}MB ({result['reduction']:.1f}%)")
        print(f" Rows: {result['rows']:,} | Columns: {result['columns_kept']}")
        stats['presidential'] = result
        stats['total_original'] += result['original_size']
        stats['total_reduced'] += result['new_size']
        stats['total_rows'] += result['rows']
    else:
        print(f" ✗ FAILED or file not found")
    
    # [2] Candidate data
    print("\n[2/3] CANDIDATE DATA (consulta_cand_2022_BRASIL.csv)")
    print("-" * 100)
    
    cand_file = CONSULTA_CAND_PATH / "consulta_cand_2022_BRASIL.csv"
    output_file = CONSULTA_CAND_PATH / "consulta_cand_2022_BRASIL_REDUCED.csv"
    print(f" Processing: {cand_file.name}")
    
    result = process_file_streaming(cand_file, output_file, COLS_CANDIDATE, "candidate")
    
    if result:
        print(f"✓")
        print(f" Original: {result['original_size']:.1f}MB | Reduced: {result['new_size']:.1f}MB ({result['reduction']:.1f}%)")
        print(f" Rows: {result['rows']:,} | Columns: {result['columns_kept']}")
        stats['candidates'] = result
        stats['total_original'] += result['original_size']
        stats['total_reduced'] += result['new_size']
        stats['total_rows'] += result['rows']
    else:
        print(f" ✗ FAILED or file not found")
    
    # [3] Deputy data - ALL STATES
    print(f"\n[3/3] DEPUTY DATA - ALL {len(UFS)} STATES")
    print("-" * 100)
    print(" NOTE: Processing with streaming. RAM usage will stay constant ~5-10MB")
    
    for idx, uf in enumerate(UFS, 1):
        uf_file = BASE_PATH / f"votacao_secao_2022_{uf}.csv"
        output_file = BASE_PATH / f"votacao_secao_2022_{uf}_REDUCED.csv"
        
        is_large = uf in ['SP', 'MG', 'BA', 'RJ', 'RS', 'PR']
        size_indicator = "LARGE " if is_large else ""
        
        print(f" [{idx:2d}/{len(UFS)}] {uf} {size_indicator}... ", end='', flush=True)
        
        result = process_file_streaming(uf_file, output_file, COLS_DEPUTY, f"state {uf}")
        
        if result:
            stats['states'][uf] = result
            stats['total_original'] += result['original_size']
            stats['total_reduced'] += result['new_size']
            stats['total_rows'] += result['rows']
            stats['successful_states'] += 1
            print(f"✓ ({result['original_size']:.0f}→{result['new_size']:.0f}MB | {result['rows']:,} rows)")
        else:
            print(f"✗ FAILED")
            stats['failed_states'].append(uf)
    
    # FINAL SUMMARY
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 100)
    print("PROCESSING COMPLETE - FINAL SUMMARY")
    print("=" * 100)
    print(f"Total Time: {format_time(elapsed_time)}")
    print(f"States Processed: {stats['successful_states']}/{len(UFS)} successful")
    print(f"Total Rows Processed: {stats['total_rows']:,}")
    print(f"Total Original Size: {stats['total_original']:.1f}MB")
    print(f"Total Reduced Size: {stats['total_reduced']:.1f}MB")
    
    if stats['total_original'] > 0:
        overall_reduction = ((stats['total_original'] - stats['total_reduced']) / stats['total_original'] * 100)
        print(f"Overall Size Reduction: {overall_reduction:.1f}%")
    
    if stats['failed_states']:
        print(f"\n⚠️ FAILED States ({len(stats['failed_states'])}): {', '.join(stats['failed_states'])}")
    else:
        print(f"\n✓ ALL STATES PROCESSED SUCCESSFULLY!")
    
    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("1. Verify output files exist and have correct columns:")
    print("   head -1 data/votacao_secao_2022_SP_REDUCED.csv")
    print("\n2. Expected columns (no CD_CARGO/PERGUNTA):")
    print("   NR_TURNO;CD_CARGO;SG_UF;CD_MUNICIPIO;NM_MUNICIPIO;NR_VOTAVEL;QT_VOTOS")
    print("\n3. Test analysis script:")
    print("   python3 analise_descritiva_simples.py")
    print("\n4. If all OK, files are ready for use!")
    print("=" * 100)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
