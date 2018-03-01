import weka.associations.Apriori;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

public class AssociationRuleLearning {
    public static void main(String[] args) {
        try {
            BufferedReader br;
            br = new BufferedReader(new FileReader("babies.arff"));
            Instances data = new Instances(br);
            br.close();

            Apriori apriori = new Apriori();
            apriori.setNumRules(100);
            apriori.setMinMetric(0.5);

            apriori.buildAssociations(data);
            System.out.println(apriori);
        } catch (Exception ex) {
// Handle exceptions
        }

    }
}
