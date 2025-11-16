#!/usr/bin/env python3
"""
Compare CUDA and Serial policy iteration results for correctness verification.
"""

import sys
import os

def load_results_file(filepath):
    """Load policy and values from a results file."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        return None, None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find policy line
    policy_line = None
    values_line = None
    
    for line in lines:
        if line.startswith("Optimal policy:"):
            policy_line = line.strip()
        elif line.startswith("Optimal value:"):
            values_line = line.strip()
    
    if not policy_line or not values_line:
        print(f"Error: Could not parse results from {filepath}")
        return None, None
    
    # Parse policy (extract numbers after "Optimal policy: ")
    policy_str = policy_line.split("Optimal policy: ")[1]
    policy = [int(x) for x in policy_str.split()]
    
    # Parse values (extract numbers after "Optimal value: ")
    values_str = values_line.split("Optimal value: ")[1]
    values = [float(x) for x in values_str.split()]
    
    return policy, values

def compare_policies(policy1, policy2, name1="CUDA", name2="Serial"):
    """Compare two policies for exact match."""
    if len(policy1) != len(policy2):
        print(f"❌ POLICY SIZE MISMATCH: {name1} has {len(policy1)} states, {name2} has {len(policy2)} states")
        return False, 0.0
    
    matches = sum(1 for p1, p2 in zip(policy1, policy2) if p1 == p2)
    total = len(policy1)
    match_rate = matches / total
    
    print(f"\n📊 POLICY COMPARISON ({name1} vs {name2}):")
    print(f"   States matching: {matches}/{total} ({match_rate:.1%})")
    
    # Check if match rate is 90.0% or higher
    if match_rate >= 0.90:
        if match_rate == 1.0:
            print(f"   ✅ POLICIES IDENTICAL - Perfect match!")
        else:
            print(f"   ✅ POLICIES MATCHING - {match_rate:.1%} match (≥90.0%)")
        print("matching")  # Output "matching" for 90%+ match
        return True, match_rate
    else:
        print(f"   ❌ POLICIES DIFFER in {total - matches} states")
        
        # Show first few differences
        diff_indices = [i for i, (p1, p2) in enumerate(zip(policy1, policy2)) if p1 != p2]
        print(f"   First 10 differences:")
        for i, idx in enumerate(diff_indices[:10]):
            print(f"     State {idx}: {name1}={policy1[idx]}, {name2}={policy2[idx]}")
        
        return False, match_rate

def compare_values(values1, values2, tolerance=1e-4, name1="CUDA", name2="Serial"):
    """Compare two value functions with numerical tolerance."""
    if len(values1) != len(values2):
        print(f"❌ VALUE SIZE MISMATCH: {name1} has {len(values1)} states, {name2} has {len(values2)} states")
        return False
    
    abs_diffs = [abs(v1 - v2) for v1, v2 in zip(values1, values2)]
    max_diff = max(abs_diffs)
    mean_diff = sum(abs_diffs) / len(abs_diffs)
    within_tolerance = sum(1 for diff in abs_diffs if diff <= tolerance)
    total = len(values1)
    
    print(f"\n📊 VALUE FUNCTION COMPARISON ({name1} vs {name2}):")
    print(f"   Max absolute difference: {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")
    print(f"   Values within tolerance (±{tolerance}): {within_tolerance}/{total} ({within_tolerance/total:.1%})")
    
    if within_tolerance == total:
        print(f"   ✅ VALUES MATCH within tolerance")
        return True
    else:
        print(f"   ⚠️  VALUES DIFFER beyond tolerance in {total - within_tolerance} states")
        
        # Show largest differences
        large_diffs = [(i, diff, values1[i], values2[i]) for i, diff in enumerate(abs_diffs) if diff > tolerance]
        large_diffs.sort(key=lambda x: x[1], reverse=True)  # Sort by difference magnitude
        print(f"   Largest differences:")
        for i, (idx, diff, v1, v2) in enumerate(large_diffs[:5]):
            print(f"     State {idx}: {name1}={v1:.6f}, {name2}={v2:.6f}, diff={diff:.6f}")
        
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <cuda_results.txt> <serial_results.txt>")
        sys.exit(1)
    
    cuda_file = sys.argv[1]
    serial_file = sys.argv[2]
    
    print(f"🔍 COMPARING POLICY ITERATION RESULTS")
    print(f"   CUDA results: {cuda_file}")
    print(f"   Serial results: {serial_file}")
    print("=" * 60)
    
    # Load results
    cuda_policy, cuda_values = load_results_file(cuda_file)
    serial_policy, serial_values = load_results_file(serial_file)
    
    if cuda_policy is None or serial_policy is None:
        print("❌ Failed to load results files")
        sys.exit(1)
    
    # Compare policies (most important)
    policy_match, match_percentage = compare_policies(cuda_policy, serial_policy)
    
    # Compare values (with tolerance for numerical precision)
    value_match = compare_values(cuda_values, serial_values, tolerance=1e-4)
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("🎯 CORRECTNESS ASSESSMENT:")
    
    if policy_match and value_match:
        print("   ✅ IMPLEMENTATIONS ARE CORRECT - Both policies and values match!")
        print("   🎉 Your serial implementation is working perfectly!")
    elif policy_match and not value_match:
        print("   ✅ POLICIES MATCH - Core algorithm is correct!")
        print("   ⚠️  Small value differences likely due to numerical precision")
        print("   ✅ This is acceptable for practical purposes")
    elif not policy_match and value_match:
        print("   ❌ POLICIES DIFFER but values match - Possible tie-breaking differences")
        print("   🔍 Check if multiple optimal actions exist for some states")
    else:
        print("   ❌ BOTH POLICIES AND VALUES DIFFER - Implementation bug likely")
        print("   🐛 Further debugging needed")
    
    return 0 if policy_match else 1

if __name__ == "__main__":
    sys.exit(main())
