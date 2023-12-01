import java.util.Arrays;

public class Main {
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
        LogisticRegression.importCSV(filePath, features, trainingFeatures, trainingLabels);
        System.out.println("trainingLabels: " + Arrays.deepToString(trainingFeatures));

        // Training the model
        logisticRegression.train(trainingFeatures, trainingLabels, epochs);

        // Sample test data
        double[] testFeatures = {-4, -6};

        // Making predictions
        int predictedClass = logisticRegression.predictClass(testFeatures);
        System.out.println("Predicted Class: " + predictedClass);
    }
}
