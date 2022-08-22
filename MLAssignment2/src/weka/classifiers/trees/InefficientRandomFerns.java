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

public class InefficientRandomFerns extends RandomizableClassifier {

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

    protected PartitionedMultiFilter m_PMF;
    protected NaiveBayes m_NB;

    @Override
    public void buildClassifier(Instances data) throws Exception {

        // Check if classifier can handle the data
        getCapabilities().testWithFail(data);

        // Make list of indices of non-class attributes and shuffle it
        List<Integer> indices = Collections.list(data.enumerateAttributes()).stream().
                map(att -> att.index()).collect(Collectors.toList());
        Collections.shuffle(indices, data.getRandomNumberGenerator(getSeed()));

        // Create sublists of indices of the chosen size
        List<int[]> listOfSublists= new ArrayList<>();
        for (int indexOfIndex = 0; indexOfIndex < indices.size(); ) {
            int next = Math.min(indexOfIndex + getSize(), indices.size());
            listOfSublists.add(indices.subList(indexOfIndex, next).stream().mapToInt(Integer::intValue).toArray());
            indexOfIndex = next;
        }

        // Set up filters to form cartesian products of the attribute sets corresponding to the ranges
        m_PMF = new PartitionedMultiFilter();
        Range[] ranges = new Range[listOfSublists.size()];
        Filter[] filters = new Filter[ranges.length];
        for (int i = 0; i < ranges.length; i++) {
            ranges[i] = new Range(Range.indicesToRangeList(listOfSublists.get(i)));
            if (listOfSublists.get(i).length >= 2) { // Don't filter if there is no product to be made!
                MultiFilter MF = new MultiFilter(); // We need to combine CartesianProduct with Remove
                CartesianProduct cp = new CartesianProduct();
                cp.setAttributeIndices("first-last");
                Remove r = new Remove();
                r.setAttributeIndices("last"); // We want to only keep the cartesian product, not the original att.
                r.setInvertSelection(true);
                Filter[] filterCombo = new Filter[2];
                filterCombo[0] = cp;
                filterCombo[1] = r;
                MF.setFilters(filterCombo);
                filters[i] = MF;
            } else {
                filters[i] = new AllFilter(); // This filter leaves the data untouched (bad name!)
            }
        }

        // We can set up and apply the PartitionedMultiFilter
        m_PMF.setFilters(filters);
        m_PMF.setRanges(ranges);
        m_PMF.setInputFormat(data);
        data = Filter.useFilter(data, m_PMF);

        // CartesianProduct gives each new attribute a weight based on how many attributes are combined: undo!
        for (int i = 0; i < data.numAttributes(); i++) { data.attribute(i).setWeight(1.0); };

        // Now simply build naive Bayes model based on the transformed set of attributes
        m_NB = new NaiveBayes();
        m_NB.buildClassifier(data);
    }

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {

        // Get class probability estimates from naive Bayes after filtering the instance
        m_PMF.input(inst);
        return m_NB.distributionForInstance(m_PMF.output());
    }

    @Override
    public String toString() { return m_NB != null ? m_NB.toString() : "Model not built yet"; }

    public static void main(String[] arguments) { runClassifier(new RandomFerns(), arguments); }
}