/*
basic neural network using standard Java libraries
 */
public class BasicNeuralNetworkStandardLibs {
    double errors;
    int inputNeurons;
    int outputNeurons;
    int hiddenNeurons;
    int totalNeurons;
    int weights;
    double learningRate;
    double outputResults[];
    double resultsMatrix[];
    double lastErrors[];
    double changes[];
    double thresholds[];
    double weightChanges[];
    double allThresholds[];
    double threshChanges[];
    double momentum;
    double errorChanges[];


    public BasicNeuralNetworkStandardLibs(int inputCount,
                                          int hiddenCount,
                                          int outputCount,
                                          double learnRate,
                                          double momentum) {
        learningRate = learnRate;
        momentum = momentum;

        inputNeurons = inputCount;
        hiddenNeurons = hiddenCount;
        outputNeurons = outputCount;
        totalNeurons = inputCount + hiddenCount + outputCount;
        weights = (inputCount * hiddenCount)
                + (hiddenCount * outputCount);

        outputResults = new double[totalNeurons];
        resultsMatrix = new double[weights];
        weightChanges = new double[weights];
        thresholds = new double[totalNeurons];
        errorChanges = new double[totalNeurons];
        lastErrors = new double[totalNeurons];
        allThresholds = new double[totalNeurons];
        changes = new double[weights];
        threshChanges = new double[totalNeurons];
        reset();

    }

    public void reset() {
        int loc;
        for (loc = 0; loc < totalNeurons; loc++) {
            thresholds[loc] = 0.5 - (Math.random());
            threshChanges[loc] = 0;
            allThresholds[loc] = 0;
        }
        for (loc = 0; loc < resultsMatrix.length; loc++) {
            resultsMatrix[loc] = 0.5 - (Math.random());
            weightChanges[loc] = 0;
            changes[loc] = 0;
        }
    }

    public double threshold(double sum) {
        return 1.0 / (1 + Math.exp(-1.0 * sum));
    }

    public double[] calcOutput(double input[]) {
        int loc, pos;
        final int hiddenIndex = inputNeurons;
        final int outIndex = inputNeurons + hiddenNeurons;
        for (loc = 0; loc < inputNeurons; loc++) {
            outputResults[loc] = input[loc];
        }

        int rLoc = 0;
        for (loc = hiddenIndex; loc < outIndex; loc++) {
            double sum = thresholds[loc];
            for (pos = 0; pos < inputNeurons; pos++) {
                sum += outputResults[pos] * resultsMatrix[rLoc++];
            }
            outputResults[loc] = threshold(sum);
        }

        double result[] = new double[outputNeurons];
        for (loc = outIndex; loc < totalNeurons; loc++) {
            double sum = thresholds[loc];
            for (pos = hiddenIndex; pos < outIndex; pos++) {
                sum += outputResults[pos] * resultsMatrix[rLoc++];
            }
            outputResults[loc] = threshold(sum);
            result[loc-outIndex] = outputResults[loc];
        }
        return result;
    }

    public void calcError(double ideal[]) {
        int loc, pos;
        final int hiddenIndex = inputNeurons;
        final int outputIndex = inputNeurons + hiddenNeurons;
        for (loc = inputNeurons; loc < totalNeurons; loc++) {
            lastErrors[loc] = 0;
        }

        for (loc = outputIndex; loc < totalNeurons; loc++) {
            lastErrors[loc] = ideal[loc - outputIndex] -
                    outputResults[loc];
            errors += lastErrors[loc] * lastErrors[loc];
            errorChanges[loc] = lastErrors[loc] * outputResults[loc]
                    * (1 - outputResults[loc]);
        }

        int locx = inputNeurons * hiddenNeurons;
        for (loc = outputIndex; loc < totalNeurons; loc++) {
            for (pos = hiddenIndex; pos < outputIndex; pos++) {
                changes[locx] += errorChanges[loc] *
                        outputResults[pos];
                lastErrors[pos] += resultsMatrix[locx] *
                        errorChanges[loc];
                locx++;
            }
            allThresholds[loc] += errorChanges[loc];
        }

        for (loc = hiddenIndex; loc < outputIndex; loc++) {
            errorChanges[loc] = lastErrors[loc] *outputResults[loc]
                    * (1 - outputResults[loc]);
        }

        locx = 0;
        for (loc = hiddenIndex; loc < outputIndex; loc++) {
            for (pos = 0; pos < hiddenIndex; pos++) {
                changes[locx] += errorChanges[loc] *
                        outputResults[pos];
                lastErrors[pos] += resultsMatrix[locx] *
                        errorChanges[loc];
                locx++;
            }
            allThresholds[loc] += errorChanges[loc];
        }
    }

    public double getError(int len) {
        double err = Math.sqrt(errors / (len * outputNeurons));
        errors = 0;
        return err;
    }

    public void train() {
        int loc;
        for (loc = 0; loc < resultsMatrix.length; loc++) {
            weightChanges[loc] = (learningRate * changes[loc]) +
                    (momentum * weightChanges[loc]);
            resultsMatrix[loc] += weightChanges[loc];
            changes[loc] = 0;
        }
        for (loc = inputNeurons; loc < totalNeurons; loc++) {
            threshChanges[loc] = learningRate * allThresholds[loc] +
                    (momentum * threshChanges[loc]);
            thresholds[loc] += threshChanges[loc];
            allThresholds[loc] = 0;
        }
    }

    public static void main(String[] args){
        double xorIN[][] ={
                {0.0,0.0},
                {1.0,0.0},
                {0.0,1.0},
                {1.0,1.0}};
        double xorEXPECTED[][] = { {0.0},{1.0},{1.0},{0.0}};

        BasicNeuralNetworkStandardLibs network = new
                BasicNeuralNetworkStandardLibs(2,3,1,0.7,0.9);

        for (int runCnt=0;runCnt<10000;runCnt++) {
            for (int loc=0;loc<xorIN.length;loc++) {
                network.calcOutput(xorIN[loc]);
                network.calcError(xorEXPECTED[loc]);
                network.train();
            }
            System.out.println("Trial #" + runCnt + ",Error:" +
                    network.getError(xorIN.length));
        }
    }

}
