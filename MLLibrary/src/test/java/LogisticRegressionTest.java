import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class LogisticRegressionTest {
    @Test
    public void testPredictFunction() {
        int features = 2;
        double learningRate = 0.01;
        int epochs = 1000;

        LogisticRegression logisticRegression = new LogisticRegression(features, learningRate);
        double tolerance = 1e-2; // Tolerance for floating-point comparisons

        // Test cases with known results
        double[] weights = {1.0, -2.0}; // Sample weights
        logisticRegression.setWeights(weights); // Set the weights for testing

        double[] features1 = {1.0, 2.0}; // Positive example
        double prediction1 = logisticRegression.predict(features1);
        assertEquals(0.0474, prediction1, tolerance);

        double[] features2 = {-1.0, -2.0}; // Negative example
        double prediction2 = logisticRegression.predict(features2);
        assertEquals(0.952, prediction2, tolerance);

    }

    @Test
    public void testLogisticRegression() {
        int features = 2;
        double learningRate = 0.01;
        int epochs = 1000;

        LogisticRegression logisticRegression = new LogisticRegression(features, learningRate);

        double tolerance = 1e-5; // Tolerance for floating-point comparisons

        // Test cases with known results
        assertEquals(0.5, logisticRegression.sigmoid(0), tolerance);
        assertEquals(1.0, logisticRegression.sigmoid(1000), tolerance);
        assertEquals(0.0, logisticRegression.sigmoid(-1000), tolerance);

        // Test cases for edge values
        assertEquals(1.0, logisticRegression.sigmoid(Double.POSITIVE_INFINITY), tolerance);
        assertEquals(0.0, logisticRegression.sigmoid(Double.NEGATIVE_INFINITY), tolerance);
        assertTrue(Double.isNaN(logisticRegression.sigmoid(Double.NaN)));

        // Define the path to your CSV file
        String filePath = "src/main/java/test.csv";

        // Define the size of your dataset (rows)
        int dataSize = 100;

        // Initialize arrays to store features and labels
        double[][] trainingFeatures = new double[dataSize][features];
        int[] trainingLabels = new int[dataSize];

        // Import data from CSV
        LogisticRegression.importCSV(filePath, features, trainingFeatures, trainingLabels);

        // Training the model
        logisticRegression.train(trainingFeatures, trainingLabels, epochs);


        // Test predictions on new data
        assertAll(() -> assertEquals(0, logisticRegression.predictClass(new double[]{0, 0})),
                () -> assertEquals(0, logisticRegression.predictClass(new double[]{-4, -2})),
                () -> assertEquals(1, logisticRegression.predictClass(new double[]{34.62365962451697, 78.0246928153624})),
                () -> assertEquals(1, logisticRegression.predictClass(new double[]{35.84740876993872, 72.90219802708364}))
        );

        // Add more tests as needed
    }
}