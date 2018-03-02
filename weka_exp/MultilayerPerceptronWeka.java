import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.FileReader;

public class MultilayerPerceptronWeka {
    public static void main(String[] args) {
        String trainingFileName = "dermatologyTrainingSet.arff";
        String testingFileName = "dermatologyTestingSet.arff";
        try (FileReader trainingReader = new FileReader(trainingFileName);
             FileReader testingReader =
                     new FileReader(testingFileName)) {
            Instances trainingInstances = new Instances(trainingReader);
            trainingInstances.setClassIndex(
                    trainingInstances.numAttributes() - 1);
            Instances testingInstances = new Instances(testingReader);
            testingInstances.setClassIndex(
                    testingInstances.numAttributes() - 1);

            MultilayerPerceptron mlp = new MultilayerPerceptron();

            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(2000);
            mlp.setHiddenLayers("3");

            mlp.buildClassifier(trainingInstances);

            Evaluation evaluation = new Evaluation(trainingInstances);
            evaluation.evaluateModel(mlp, testingInstances);

            System.out.println(evaluation.toSummaryString());

            for (int i = 0; i < testingInstances.numInstances(); i++) {
                double result = mlp.classifyInstance(
                        testingInstances.instance(i));
                if (result != testingInstances
                        .instance(i)
                        .value(testingInstances.numAttributes() - 1)) {
                    System.out.println("Classify result: " + result
                            + " Correct: " + testingInstances.instance(i)
                            .value(testingInstances.numAttributes() - 1));

                    Instance incorrectInstance = testingInstances.instance(i);
                    incorrectInstance.setDataset(trainingInstances);
                    double[] distribution = mlp.distributionForInstance(incorrectInstance);
                    System.out.println("Probability of being positive: " + distribution[0]);
                    System.out.println("Probability of being negative: " + distribution[1]);
                }
            }

            SerializationHelper.write("mlpModel", mlp);
            //mlp = (MultilayerPerceptron)SerializationHelper.read("mlpModel");
        } catch (Exception ex) {
// Handle exceptions
        }
    }
}
