import java.util.*;
import java.io.*;

public class LinearRegression {

    // arraylist to store the coordinates - pairwise x, y
    ArrayList<double[]> al = new ArrayList<double[]>();
    ArrayList<double[]> train_points = new ArrayList<double[]>(); // to store training data points
    ArrayList<double[]> test_points = new ArrayList<double[]>(); // to store testing data points
    static double avg_x = 0, avg_y = 0; // variables to store the means
    double sd_x = 0, sd_y = 0; // variables to store the standard deviations
    static double sum_x = 0, sum_y = 0, sum_XY = 0; // store the sums of numbers
    static double sum_X_sq = 0, sum_Y_sq = 0;
    double coef_ = 0, slope_ = 0; // store the calculated slope and coefficient of correlation
    double intercept_ = 0; // stores the intercept value for the equation
    double mean_squared_error_ = 0; // stores the mean squared error value
    double r_square_score = 0; // stores the r-square score
    double sum_squared_resid = 0; // stores squared sum of residuals

    void calcMeans() {
        for (double[] coord : this.train_points) {
            // calculates the sum value for the variables
            sum_x += coord[0];
            sum_y += coord[1];
        }
        avg_x = sum_x / train_points.size();
        avg_y = sum_y / train_points.size();
    }
    void calcStandardDeviation() {
        // nothing to calculate if there are no datapoints and end the program
        if (train_points.isEmpty()) {
            System.out.println("No datapoints to calculate regression");
            System.exit(0);
        }

        // iterate through the list of datapoints to calculate sum squares of
        for (double[] coord : this.train_points) {
            sum_X_sq += (coord[0] - avg_x) * (coord[0] - avg_x);
            sum_Y_sq += (coord[1] - avg_y) * (coord[1] - avg_y);
            sum_XY += (coord[0] - avg_x) * (coord[1] - avg_y);
        }

        this.sd_x = Math.pow(sum_X_sq / train_points.size(), 0.5);
        this.sd_y = Math.pow(sum_Y_sq / train_points.size(), 0.5);

    }

    void calcCorrelation() {
        // calculates coefficient of correlation
        this.coef_ = sum_XY / (this.train_points.size() * sd_x * sd_y);

        // calculates m and c values for the equation y=mx+c
        this.slope_ = coef_ * sd_y / sd_x;
        this.intercept_ = avg_y - (this.slope_ * avg_x);
    }

    void displayResults() {

        if (this.coef_ > 0.5 || this.coef_ < -0.5) {
            System.out.println("Well co-related data. Worth creating the linear model");
        } else {
            System.out.println("Not very co-related data.");
        }

        System.out.println("Coefficient of correlation : " + this.coef_ + "\n");
        System.out.println("Linear Model created with equation : y=" + this.slope_ + "x+" + this.intercept_ + "\n");
        System.out.println("MSE score for the model : " + this.mean_squared_error_ + "\n");

        if (this.r_square_score > 0.5) {
            System.out.println("Very good R^2 score.");
        } else {
            System.out.println("R^2 score not very good for model creation");
        }
        System.out.println("R Square Score for the model : " + this.r_square_score);
    }

    void fit() {
        // calculates all the required values for the model
        this.train_test_split();
        this.calcMeans();
        this.calcStandardDeviation();
        this.calcCorrelation();
        this.calcMeanSquaredError();
        this.calcRSquareScore();
        this.displayResults();
    }

    void train_test_split() {
        int num_train_points = (int) Math.floor(0.7 * al.size());
        this.test_points = new ArrayList<>(this.al);
        for (int i = 0; i < num_train_points; i++) {
            double[] pt = al.get((int) Math.floor(Math.random() * num_train_points));
            this.train_points.add(pt);
            this.test_points.remove(pt);
        }
    }

    double predict(double X) {
        // predicts the value of dependent variable on the basis of the X value
        double Y = this.slope_ * X + this.intercept_;
        return Y;
    }

    void calcMeanSquaredError() {
        // finds sum of squares of errors
        this.sum_squared_resid = 0;
        for (double[] coord : this.test_points) {
            this.sum_squared_resid += Math.pow(coord[1] - (this.slope_ * coord[0] + this.intercept_), 2);
        }
        // finds mean of squared errors
        this.mean_squared_error_ = sum_squared_resid / this.test_points.size();
    }

    void calcRSquareScore() {
        // finds R^2 score
        this.r_square_score = 1 - (this.sum_squared_resid) / (sum_Y_sq);
    }

    public void loadCSV(String filename) throws IOException {

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {

                String[] data = line.split(",");
                double[] new_coord = { Double.parseDouble(data[0]), Double.parseDouble(data[1]) };
                al.add(new_coord);

            }
        }
    }

    public static void main(String[] args) {

        try {
            LinearRegression lm = new LinearRegression();

            lm.loadCSV("src/main/java/testLR.csv");
            lm.fit();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
