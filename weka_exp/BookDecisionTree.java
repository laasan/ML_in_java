import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class BookDecisionTree {
    private Instances trainingData;

    public static void main(String[] args) {
        try {
            BookDecisionTree decisionTree =
                    new BookDecisionTree("books.arff");
            J48 tree = decisionTree.performTraining();
            System.out.println(tree.toString());
        } catch (Exception ex) {
// Handle exceptions
        }
    }


    public BookDecisionTree(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(
                    new FileReader(fileName));
            trainingData = new Instances(reader);
            trainingData.setClassIndex(
                    trainingData.numAttributes() - 1);
        } catch (IOException ex) {
// Handle exceptions
        }
    }

    private J48 performTraining() {
        J48 j48 = new J48();
        String[] options = {"-U"};
        try {
            j48.setOptions(options);
            j48.buildClassifier(trainingData);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return j48;
    }

    private Instance getTestInstance(
            String binding, String multicolor, String genre) {
        Instance instance = new DenseInstance(3);
        instance.setDataset(trainingData);
        instance.setValue(trainingData.attribute(0), binding);
        instance.setValue(trainingData.attribute(1), multicolor);
        instance.setValue(trainingData.attribute(2), genre);
        return instance;
    }

}