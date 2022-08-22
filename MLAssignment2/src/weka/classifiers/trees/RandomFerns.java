package weka.classifiers.trees;

import weka.classifiers.RandomizableClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;
import weka.filters.AllFilter;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.CartesianProduct;
import weka.filters.unsupervised.attribute.PartitionedMultiFilter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import java.io.*;
import java.util.*;
public class RandomFerns extends RandomizableClassifier {

    protected int m_Size = 1;
    @OptionMetadata(displayName = "size", description = "Size of fern (default = 1).",
            commandLineParamName = "size", commandLineParamSynopsis = "-size <int>", displayOrder = 1)
    public int getSize() { return m_Size; }
    public void setSize(int size) { this.m_Size = size; }

    public Capabilities getCapabilities() {

        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        return result;
    }

    protected List<Hashtable> hashTables = new ArrayList<>();
    protected int numInstances = 0;
    protected List<int[]> listOfSublists = new ArrayList<>();
    protected int[] classNumbers;
    protected Instances storedData;

    @Override
    public void buildClassifier(Instances data) throws Exception {

        numInstances = data.numInstances();
        storedData = data.stringFreeStructure();
        classNumbers = new int[data.classAttribute().numValues()];
        Arrays.fill(classNumbers, 0);

        // Check if classifier can handle the data
        getCapabilities().testWithFail(data);

        // Make list of indices of non-class attributes and shuffle it
        List<Integer> indices = Collections.list(data.enumerateAttributes()).stream().
                map(att -> att.index()).collect(Collectors.toList());
        Collections.shuffle(indices, data.getRandomNumberGenerator(getSeed()));

        // Create sublists of indices of the chosen size
        for (int indexOfIndex = 0; indexOfIndex < indices.size(); ) {
            int next = Math.min(indexOfIndex + getSize(), indices.size());
            listOfSublists.add(indices.subList(indexOfIndex, next).stream().mapToInt(Integer::intValue).toArray());
            indexOfIndex = next;
        }

        // Create hash tables for data
        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            Hashtable<String, String> ht = new Hashtable<>();
            hashTables.add(ht);
        }

        // For each instance
        for (int i = 0; i < data.numInstances(); i++) {
            // Get current instance
            Instance current = data.instance(i);
            //System.out.println("Current Instance: " + current);

            // Get class attribute index
            int classIndex = (int)current.classValue();
            Range[] ranges = new Range[listOfSublists.size()];
            // For each sublist
            for (int j = 0; j < ranges.length; j++) {
                ranges[j] = new Range(Range.indicesToRangeList(listOfSublists.get(j)));
                // For each sublist attribute
                String hashKey = "";
                for (int k = 0; k < listOfSublists.get(j).length; k++) {
                    int attributeIndex = listOfSublists.get(j)[k];
                    // Add attribute value to hash key
                    if (k == 0) {
                        hashKey = current.stringValue(attributeIndex) + "/" + attributeIndex;
                    }
                    else {
                        hashKey = hashKey +  "--" + current.stringValue(attributeIndex) + "/" + attributeIndex;
                    }
                }
                //System.out.println("Hash Key: " + hashKey);

                // Check if hash table contains hash key
                Hashtable currentTable = hashTables.get(classIndex);
                if (currentTable.containsKey(hashKey)) {
                    // If contains hash key update value by 1
                    currentTable.replace(hashKey, currentTable.get(hashKey), (int)currentTable.get(hashKey) + 1);
                }
                else {
                    // Else add hash key to table with initial value of 1
                    currentTable.put(hashKey, 1);
                }
            }

            // Add to value in class list
            classNumbers[classIndex] ++;
        }
    }

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {

        // Array for class value probabilities
        double[] probabilities = new double[hashTables.size()];

        // Loop through hash tables to initialise probabilities array
        for (int a = 0; a < hashTables.size(); a++) {
            // Set probabilities array with class probability values
            probabilities[a] = ((double) classNumbers[a] + 1) / ((double) numInstances + storedData.numClasses());
        }

        // For each sublist
        for (int x = 0; x < listOfSublists.size(); x++) {
            String hashKey = "";
            int n = 0;
            // For each sublist attribute
            for (int y = 0; y < listOfSublists.get(x).length; y++) {
                int attributeIndex = listOfSublists.get(x)[y];
                // Add attribute value to hash key
                if (y == 0) {
                    hashKey = hashKey + inst.stringValue(attributeIndex) + "/" + attributeIndex;
                }
                else {
                    hashKey = hashKey +  "--" + inst.stringValue(attributeIndex) + "/" + attributeIndex;
                }
                // Calculate number of possible attributes
                if (n == 0) {
                    n = storedData.attribute(attributeIndex).numValues();
                }
                else {
                    n = n * storedData.attribute(attributeIndex).numValues();
                }
            }

            // Loop through hash tables
            for (int hashtableIndex = 0; hashtableIndex < hashTables.size(); hashtableIndex++) {
                // Check if hash table has hash key in it
                if (hashTables.get(hashtableIndex).containsKey(hashKey)) {
                    int count = (int) hashTables.get(hashtableIndex).get(hashKey);

                    // Calculate subset probability with laplace estimator
                    double subsetProb = ((double) count + 1) / ((double) classNumbers[hashtableIndex] + (double) n);
                    // Add into probabilities list
                    probabilities[hashtableIndex] = probabilities[hashtableIndex] * subsetProb;
                }
                else {
                    // Calculate subset probability with laplace estimator
                    double subsetProb = (1) / ((double) classNumbers[hashtableIndex] + (double) n);
                    // Add into probabilities list
                    probabilities[hashtableIndex] = probabilities[hashtableIndex] * subsetProb;
                }
            }
        }

        // Return normalized array
        Utils.normalize(probabilities);
        return probabilities;
    }

    @Override
    public String toString() {

        String returnString = "Class " + storedData.classAttribute().toString() + '\n';

        // If hash tables aren't built yet
        if (hashTables.size() == 0) {
            returnString =  "Hashtable(s) not built yet";
        }
        // Else print all hash tables
        else {
            for (int n = 0; n < hashTables.size(); n++) {
                // Creating Enumeration interface
                Enumeration<String> e = hashTables.get(n).keys();

                if (n != 0) {
                    returnString += '\n' + "";
                }

                returnString += '\n' + "--------------------------------------------------------";
                returnString += '\n' + "Hashtable " + (n + 1) + '\n' + "Class Value: " + storedData.classAttribute().value(n);
                returnString += '\n' + "--------------------------------------------------------";
                returnString += '\n' + "Attribute Combination: Count";
                // Checking for next element in Hashtable object
                while (e.hasMoreElements()) {
                    // Getting the key of a particular entry
                    String key = e.nextElement();

                    // Print and display the Rank and Name
                    returnString += '\n' + String.format("%20s", key).replace("--", " ") + ": " + "\t"  + hashTables.get(n).get(key);

                }
                returnString += '\n' + "Total Class Number: " + classNumbers[n];
            }
        }
        return returnString;
    }

    public static void main(String[] arguments) { runClassifier(new RandomFerns(), arguments); }
}