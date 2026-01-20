#!/usr/bin/env python3
"""
Cross-validation script for Original CSTA: Alternates between training and inference 10 times
Saves only Split and Final results to txt file, grouped by dataset
"""

import os
import subprocess
import time
import datetime
from pathlib import Path

def run_command_quiet(cmd):
    """Run a command quietly and return success status and output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def extract_inference_results(inference_output):
    """Extract Split and Final results from inference output, grouped by dataset"""
    lines = inference_output.strip().split('\n')
    results = []
    final_results = []
    current_dataset = None
    dataset_splits = []
    
    for line in lines:
        line = line.strip()
        
        # Check for Split results
        if '[Split' in line and 'Kendall:' in line and 'Spear:' in line:
            dataset_splits.append(line)
        
        # Check for Final results (indicates end of dataset)
        elif '[FINAL -' in line and 'Kendall:' in line and 'Spear:' in line:
            # Extract dataset name from FINAL line
            if 'SumMe' in line:
                current_dataset = 'SumMe'
            elif 'TVSum' in line:
                current_dataset = 'TVSum'
            
            # Add dataset header and splits
            if current_dataset and dataset_splits:
                results.append(f"{current_dataset} Dataset:")
                results.extend(dataset_splits)
                results.append(line)  # Final result
                results.append("")  # Empty line for separation
                
                # Store final result for summary
                final_results.append((current_dataset, line))
                
            # Reset for next dataset
            dataset_splits = []
            current_dataset = None
    
    return results, final_results

def main():
    """Main validation loop"""
    print("Original CSTA Cross Validation")
    print("=" * 50)
    
    # Create results directory
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Results file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"original_csta_results_{timestamp}.txt"
    
    successful_runs = 0
    all_final_results = []  # Store all final results for summary
    
    with open(results_file, 'w') as f:
        f.write(f"Original CSTA Validation Results\n")
        f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for iteration in range(1, 11):
            print(f"Iteration {iteration}/10", end=" - ")
            f.write(f"Iteration {iteration}/10\n")
            f.write("-" * 20 + "\n")
            
            # Training phase (quiet)
            print("Training...", end=" ")
            train_success, _, train_err = run_command_quiet(
                "python3 train.py --epochs 100 --model_name GoogleNet_Attention"
            )
            
            if not train_success:
                print("FAILED (Training)")
                f.write(f"Training failed\n\n")
                continue
            
            # Inference phase
            print("Inference...", end=" ")
            inference_success, inference_out, inference_err = run_command_quiet(
                "python3 inference.py --model_name GoogleNet_Attention"
            )
            
            if not inference_success:
                print("FAILED (Inference)")
                f.write(f"Inference failed\n\n")
                continue
            
            # Extract and save results
            results, final_results = extract_inference_results(inference_out)
            
            if results:
                print("SUCCESS")
                for result in results:
                    f.write(result + "\n")
                successful_runs += 1
                
                # Store final results with iteration number
                for dataset, final_line in final_results:
                    all_final_results.append((iteration, dataset, final_line))
            else:
                print("NO RESULTS")
                f.write("No results found\n")
            
            f.write("\n")
            f.flush()
        
        # Final summary section
        if all_final_results:
            print("\nGenerating FINAL results summary...")
            f.write("=" * 50 + "\n")
            f.write("FINAL RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Group by dataset
            datasets = ['SumMe', 'TVSum']
            for dataset in datasets:
                dataset_finals = [(iter_num, final_line) for iter_num, ds, final_line in all_final_results if ds == dataset]
                
                if dataset_finals:
                    f.write(f"{dataset} Dataset - All Iterations:\n")
                    f.write("-" * 30 + "\n")
                    for iter_num, final_line in dataset_finals:
                        f.write(f"Iteration {iter_num}: {final_line}\n")
                    f.write("\n")
        
        # Final summary
        f.write("=" * 50 + "\n")
        f.write(f"Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successful runs: {successful_runs}/10\n")
    
    print(f"Completed: {successful_runs}/10 successful runs")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main() 