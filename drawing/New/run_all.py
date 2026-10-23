"""Run every analysis with default arguments.

  python run_all.py                      # defaults (TIER_6; scalability uses TIER_6/12/18/30)
  python run_all.py --tier TIER_6 --smooth 5
"""
import sys

import performance_analysis
import predictor_training_analysis
import scalability_analysis
import sensitivity_analysis
import training_analysis

if __name__ == "__main__":
    extra = sys.argv[1:]
    for mod in (performance_analysis, training_analysis, sensitivity_analysis,
                predictor_training_analysis):
        mod.main(extra)
    # scalability sweeps over tiers itself; use `python scalability_analysis.py --tiers ...` to change them
    scalability_analysis.main([])
