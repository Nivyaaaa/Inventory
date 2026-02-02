#!/usr/bin/env python
"""
Inventory Optimization Workflow - Main Entry Point
===================================================

This script orchestrates the complete inventory optimization workflow:
1. Train demand forecasting model on 2024 sales data
2. Generate demand forecasts for next 90 days
3. Optimize inventory levels
4. Save all outputs as JSON files
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.run_inventory_optimization import InventoryOptimizationWorkflow


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("INVENTORY OPTIMIZATION WORKFLOW")
    print("="*80)
    print("\nThis workflow will:")
    print("  1. Train demand forecasting model on 2024 sales data")
    print("  2. Generate demand forecasts for the next 90 days")
    print("  3. Optimize inventory levels based on forecasted demand")
    print("  4. Save all outputs as JSON files in data/outputs/")
    print("\n" + "="*80 + "\n")
    
    try:
        # Create and run workflow
        workflow = InventoryOptimizationWorkflow()
        results = workflow.run()
        
        # Print final summary
        print("\n" + "="*80)
        print("WORKFLOW EXECUTION SUMMARY")
        print("="*80)
        print(f"\nStatus: {results.get('overall_status', 'unknown').upper()}")
        
        if results['overall_status'] == 'completed':
            print("\n[OK] All steps completed successfully!")
            print("\nOutput files created:")
            output_files = results['steps']['step4_save_outputs'].get('files_created', [])
            for file in output_files:
                print(f"  • {file}")
            
            print("\nKey Results:")
            opt_results = results['steps']['step3_optimize_inventory']
            if opt_results['status'] == 'success':
                print(f"  • Total inventory units optimized: {opt_results['total_units']:.2f}")
                print(f"  • 90-day holding cost: ${opt_results['total_cost_90days']:.2f}")
                print(f"  • SKUs optimized: {opt_results['recommendations_count']} with recommendations")
        else:
            print("\n[ERROR] Workflow encountered errors. Check logs above.")
        
        print("\n" + "="*80 + "\n")
        
        return 0 if results['overall_status'] == 'completed' else 1
        
    except Exception as e:
        print(f"\n[FATAL] {str(e)}")
        print("\nCheck the error messages above for details.")
        print("="*80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
