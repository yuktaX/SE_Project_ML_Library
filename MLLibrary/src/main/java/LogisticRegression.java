import java.util.Arrays;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Random;

public class LogisticRegression {
    private static double[] weights;
    private static double learningRate;

    public LogisticRegression(int features, double learningRate) {
        this.weights = new double[features];
        this.learningRate = learningRate;
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double predict(double[] features) {
        double z = 0.0;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * features[i];
        }
        return sigmoid(z);
    }

    public static void updateWeights(double[] features, double error) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error * features[i];
        }
    }

    public static void train(double[][] features, int[] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < features.length; i++) {
                double prediction = predict(features[i]);
                double error = labels[i] - prediction;
                updateWeights(features[i], error);
            }
        }
    }

    public static int predictClass(double[] features) {
        double prediction = predict(features);
        return prediction > 0.5 ? 1 : 0;
    }

    public static void importCSV(String filePath, int numFeatures, double[][] features, int[] labels) {
        try (Reader reader = new FileReader(filePath);
             CSVParser csvParser = CSVFormat.DEFAULT.parse(reader)) {

            int row = 0;
            for (CSVRecord csvRecord : csvParser) {
                // System.out.println("CSV Record: " + csvRecord); // Debugging output
                if (csvRecord.size() == numFeatures + 1) { // Assuming the last column is the label
                    for (int i = 0; i < numFeatures; i++) {
                        features[row][i] = Double.parseDouble(csvRecord.get(i));
                    }
                    labels[row] = Integer.parseInt(csvRecord.get(numFeatures));
                    row++;
                } else {
                    System.out.println("Skipping invalid row: " + csvRecord);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void trainTestSplit(double[][] allFeatures, int[] allLabels, double[][] trainFeatures, int[] trainLabels, double[][] testFeatures, int[] testLabels, double testFraction) {
        int dataSize = allFeatures.length;
        int trainSize = (int) (dataSize * (1.0 - testFraction));
        int testSize = dataSize - trainSize;

        // Shuffle the data indices
        int[] indices = new int[dataSize];
        for (int i = 0; i < dataSize; i++) {
            indices[i] = i;
        }
        shuffleArray(indices);

        // Fill training set
        for (int i = 0; i < trainSize; i++) {
            int index = indices[i];
            System.arraycopy(allFeatures[index], 0, trainFeatures[i], 0, allFeatures[index].length);
            trainLabels[i] = allLabels[index];
        }

        // Fill test set
        for (int i = trainSize; i < dataSize; i++) {
            int index = indices[i];
            System.arraycopy(allFeatures[index], 0, testFeatures[i - trainSize], 0, allFeatures[index].length);
            testLabels[i - trainSize] = allLabels[index];
        }
    }

    // Helper method to shuffle an array of indices
    public static void shuffleArray(int[] array) {
        Random rand = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    public static void main(String[] args) {
        int features = 2;
        double learningRate = 0.01;
        int epochs = 1000;

        LogisticRegression logisticRegression = new LogisticRegression(features, learningRate);

        // Define the path to your CSV file
        String filePath = "src/main/java/test.csv";

        // Define the size of your dataset (rows)
        int dataSize = 100;

        // Initialize arrays to store features and labels
        double[][] trainingFeatures = new double[dataSize][features];
        int[] trainingLabels = new int[dataSize];

        // Import data from CSV
        importCSV(filePath, features, trainingFeatures, trainingLabels);
        System.out.println("trainingLabels: " + Arrays.deepToString(trainingFeatures));

        // Training the model
        logisticRegression.train(trainingFeatures, trainingLabels, epochs);

        // Sample test data
        double[] testFeatures = {60.45555629271532, 42.50840943572217};

        // Making predictions
        int predictedClass = logisticRegression.predictClass(testFeatures);
        System.out.println("Predicted Class: " + predictedClass);
    }
}