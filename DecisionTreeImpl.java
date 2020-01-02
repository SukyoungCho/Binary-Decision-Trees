import java.util.List;
import java.util.ArrayList;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 */
public class DecisionTreeImpl {
	public DecTreeNode root;
	public List<List<Integer>> trainData;
	public int maxPerLeaf;
	public int maxDepth;
	public int numAttr;

	// Build a decision tree given a training set
	DecisionTreeImpl(List<List<Integer>> trainDataSet, int mPerLeaf, int mDepth) {
		this.trainData = trainDataSet;
		this.maxPerLeaf = mPerLeaf;
		this.maxDepth = mDepth;
		if (this.trainData.size() > 0)
			this.numAttr = trainDataSet.get(0).size() - 1;
		this.root = buildTree(this.trainData, 0);
	}

	private DecTreeNode buildTree(List<List<Integer>> dataset, int curDepth) {
		// TODO: add code here

		DecTreeNode node = null;

		/*
		 * if empty(examples) then return default-label if all the data has the same
		 * label, return the leaf with majority label if the depth is equal to the
		 * “maximum depth” (the root node has depth 0) if the number of instances
		 * belonging to that node <= “maximum instances per leaf” if maximum info gain
		 * is 0
		 */

		// Counter for instances with label 0 and 1
		int oneCount = 0;
		int zeroCount = 0;
		for (List<Integer> data : dataset) {
			if (data.get(data.size() - 1) == 1) {
				oneCount += 1;
			} else {
				zeroCount += 1;
			}
		}

		// In conditions described above, return the leaf with majority label
		// Attributes and Threshold are set as -1
		// If # of '1's == # of '0's, then we set its lable as '1'
		if ((zeroCount + oneCount) <= this.maxPerLeaf) {
			// when <= max instances
			if (zeroCount > oneCount) {
				node = new DecTreeNode(0, -1, -1);
			} else {
				node = new DecTreeNode(1, -1, -1);
			}
		} else if (curDepth >= this.maxDepth) {
			// when it reached the maxDepth
			if (zeroCount > oneCount) {
				node = new DecTreeNode(0, -1, -1);
			} else {
				node = new DecTreeNode(1, -1, -1);
			}
		} else if (zeroCount == 0) {
			// when empty
			node = new DecTreeNode(1, -1, -1);
		} else if (oneCount == 0) {
			node = new DecTreeNode(0, -1, -1);
		} else {
			// If the maximum Information gain == 0, it is a leaf node
			// Else, get the best attribute and threshold
			attributePair bestPair = getBestPair(dataset);
			if (bestPair == null) {
				if (zeroCount > oneCount) {
					node = new DecTreeNode(0, -1, -1);
				} else {
					node = new DecTreeNode(1, -1, -1);
				}
			} else {
				// Split into left and right child nodes
				int attribute = bestPair.attribute;
				int threshold = bestPair.threshold;

				List<List<Integer>> left = new ArrayList<>();
				List<List<Integer>> right = new ArrayList<>();
				for (List<Integer> data : dataset) {
					if (data.get(attribute) <= threshold) {
						left.add(data);
					} else if (data.get(attribute) > threshold) {
						right.add(data);
					}
				}
				DecTreeNode leftChild = buildTree(left, curDepth + 1);
				DecTreeNode rightChild = buildTree(right, curDepth + 1);

				node = new DecTreeNode(-1, attribute, threshold);
				node.left = leftChild;
				node.right = rightChild;
			}
		}

		return node;
	}

	public int classify(List<Integer> instance) {
		// TODO: add code here
		// Note that the last element of the array is the label.
		return classify(this.root, instance);
	}

	/**
	 * Recursive method to classify an instance
	 * 
	 * @param node
	 * @param instance
	 * @return
	 */
	private int classify(DecTreeNode node, List<Integer> instance) {
		int defaultClass = -1;
		// if the node is leaf, return the label
		// else, traverse the tree
		if (node.isLeaf()) {
			return node.classLabel;
		} else {
			if (instance.get(node.attribute) <= node.threshold) {
				return classify(node.left, instance);
			} else if (instance.get(node.attribute) > node.threshold) {
				return classify(node.right, instance);
			}
		}
		return defaultClass;
	}

	/**
	 * Method to get the best attribute and threshold to split the data
	 * 
	 * @param dataset
	 * @return a pair of the best attribute and its threshold
	 */
	private attributePair getBestPair(List<List<Integer>> dataset) {
		attributePair bestPair = null;

		double initialEn = initialEntropy(dataset);
		double maxInfoGain = -1;
		int attribute = 0;

		for (attribute = 0; attribute < this.numAttr; attribute++) {
			// Valid thresholds are from 1 to 10 - no zero
			for (int threshold = 1; threshold <= 10; threshold++) {
				attributePair pair = new attributePair(attribute, threshold);
				double infoGain = 0;
				infoGain = calInfoGain(dataset, initialEn, pair);

				if (infoGain > maxInfoGain) {
					maxInfoGain = infoGain;
					bestPair = pair;
				}
			}
		}
		if (maxInfoGain == 0) {
			return null;
		}
		return bestPair;
	}

	/**
	 * Method to calcualte the information gain for the given attribute and
	 * threshold
	 * 
	 * @param dataset
	 * @param initialEntropy
	 * @param attributePair
	 * @return the information gain for the given attribute pair
	 */
	private double calInfoGain(List<List<Integer>> dataset, double initialEn, attributePair attributePair) {
		// counts for 0|0, 0|1, 1|0, 1|1
		int zeroZero = 0;
		int zeroOne = 0;
		int oneZero = 0;
		int oneOne = 0;
		int attribute = attributePair.attribute;
		int threshold = attributePair.threshold;

		for (List<Integer> data : dataset) {
			if (data.get(attribute) <= threshold) {
				if (data.get(data.size() - 1) == 0) {
					zeroZero++;
				} else if (data.get(data.size() - 1) == 1) {
					zeroOne++;
				}
			} else if (data.get(attribute) > threshold) {
				if (data.get(data.size() - 1) == 0) {
					oneZero++;
				} else if (data.get(data.size() - 1) == 1) {
					oneOne++;
				}
			}
		}

		double entropyLeft = calculateEntropy(zeroZero, zeroOne);
		double entropyRight = calculateEntropy(oneZero, oneOne);
		double infoGain = informationGain(initialEn, zeroZero, zeroOne, oneZero, oneOne, entropyLeft, entropyRight);
		return infoGain;

	}

	/**
	 * Helper method to calcualte the information gain.
	 *
	 * @param initialEntropy
	 * @param zeroZero
	 * @param zeroOne
	 * @param oneZero
	 * @param oneOne
	 * @return the information gain with given values
	 */
	private double informationGain(double initialEntropy, int zeroZero, int zeroOne, int oneZero, int oneOne,
			double entropyLeft, double entropyRight) {
		int total = zeroZero + zeroOne + oneZero + oneOne;
		double finalEntropy = ((double) ((double) zeroZero + (double) zeroOne) / ((double) total)) * (entropyLeft)
				+ ((double) ((double) oneZero + (double) oneOne) / ((double) total)) * (entropyRight);
		double informationGain = initialEntropy - finalEntropy;
		return informationGain;
	}
	
	/**
	 * Method to calculate the initial Entropy
	 *
	 * @param dataset
	 * @return the initial entropy
	 */
	private double initialEntropy(List<List<Integer>> dataset) {
		double initialEn = 0;
		int zeroCount = 0;
		int oneCount = 0;

		for (List<Integer> data : dataset) {
			if (data.get(data.size() - 1) == 0) {
				zeroCount += 1;
			} else if (data.get(data.size() - 1) == 1) {
				oneCount += 1;
			}
		}

		initialEn = calculateEntropy(zeroCount, oneCount);
		return initialEn;
	}
	
	/**
	 * Calcualte the Entropy
	 *
	 * @param zeroCount
	 * @param oneCount
	 * @return the entropy
	 */
	private double calculateEntropy(int zeroCount, int oneCount) {
		double entropy = 0;
		double zeroFraction = 0;
		double oneFraction = 0;
		double zeroFractionLog = 0;
		double oneFractionLog = 0;

		// We calculate Entropy value with double not int.
		// Convention is tht 0*log_2(0) = 0

		if (zeroCount != 0) {
			zeroFraction = (double) ((double) (zeroCount)) / ((double) (zeroCount + oneCount));
			zeroFractionLog = Math.log(zeroFraction) / Math.log((double) 2);
		}

		if (oneCount != 0) {
			oneFraction = (double) ((double) (oneCount)) / ((double) (zeroCount + oneCount));
			oneFractionLog = Math.log(oneFraction) / Math.log((double) 2);
		}

		entropy = -(zeroFraction * zeroFractionLog + oneFraction * oneFractionLog);
		return entropy;
	}

	// Print the decision tree in the specified format
	public void printTree() {
		printTreeNode("", this.root);
	}

	public void printTreeNode(String prefixStr, DecTreeNode node) {
		String printStr = prefixStr + "X_" + node.attribute;
		System.out.print(printStr + " <= " + String.format("%d", node.threshold));
		if (node.left.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.left.classLabel));
		} else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%d", node.threshold));
		if (node.right.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.right.classLabel));
		} else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}
	}

	public double printTest(List<List<Integer>> testDataSet) {
		int numEqual = 0;
		int numTotal = 0;
		for (int i = 0; i < testDataSet.size(); i++) {
			int prediction = classify(testDataSet.get(i));
			int groundTruth = testDataSet.get(i).get(testDataSet.get(i).size() - 1);
			System.out.println(prediction);
			if (groundTruth == prediction) {
				numEqual++;
			}
			numTotal++;
		}
		double accuracy = numEqual * 100.0 / (double) numTotal;
		System.out.println(String.format("%.2f", accuracy) + "%");
		return accuracy;
	}
}

/**
 * Separate class to store the attirbute and its threshold pair
 */
class attributePair {

	public int attribute;
	public int threshold;

	public attributePair(int x, int y) {
		this.attribute = x;
		this.threshold = y;
	}

}