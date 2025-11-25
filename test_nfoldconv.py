
import numpy as np
from hw1 import NFoldConv

def test_nfoldconv():
    # Define a simple distribution P: values {1, 2} with prob {0.5, 0.5}
    P = np.array([[1, 2], [0.5, 0.5]])
    n = 2
    
    print(f"Testing NFoldConv with P={P} and n={n}")
    
    try:
        Q = NFoldConv(P, n)
        print("Result Q:")
        print(Q)
        
        # Expected result for sum of 2 independent variables with P
        # 1+1=2 (0.25), 1+2=3 (0.25), 2+1=3 (0.25), 2+2=4 (0.25)
        # Q should be values [2, 3, 4] with probs [0.25, 0.5, 0.25]
        
        expected_values = [2, 3, 4]
        expected_probs = [0.25, 0.5, 0.25]
        
        # Check values
        if np.allclose(Q[0], expected_values) and np.allclose(Q[1], expected_probs):
            print("Test PASSED: Result matches expected distribution.")
        else:
            print("Test FAILED: Result does not match expected distribution.")
            print(f"Expected values: {expected_values}")
            print(f"Expected probs: {expected_probs}")
            
    except Exception as e:
        print(f"Test FAILED with error: {e}")

if __name__ == "__main__":
    test_nfoldconv()
