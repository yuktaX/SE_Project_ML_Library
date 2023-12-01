import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LinearRegressionTest {

    @Test
    public void testCalcMeans() {
        LinearRegression lr = new LinearRegression();
        lr.train_points.add(new double[] { 1, 2 });
        lr.train_points.add(new double[] { 3, 4 });
        lr.calcMeans();
        assertEquals(2.0, LinearRegression.avg_x, 0.001);
        assertEquals(3.0, LinearRegression.avg_y, 0.001);
    }

    @Test
    public void testCalcStandardDeviation() {
        LinearRegression lr = new LinearRegression();
        lr.train_points.add(new double[] { 1, 2 });
        lr.train_points.add(new double[] { 3, 4 });
        lr.calcMeans();
        lr.calcStandardDeviation();
        assertEquals(1.0, lr.sd_x, 0.001);
        assertEquals(1.0, lr.sd_y, 0.001);
    }

    @Test
    public void testCalcCorrelation() {
        LinearRegression lr = new LinearRegression();
        lr.train_points.add(new double[] { 1, 2 });
        lr.train_points.add(new double[] { 3, 4 });
        lr.calcMeans();
        lr.calcStandardDeviation();
        lr.calcCorrelation();
        assertEquals(1.0, lr.coef_, 0.001);
        assertEquals(1.0, lr.slope_, 0.001);
        assertEquals(1.0, lr.intercept_, 0.001);
    }

    @Test
    public void testTrainTestSplit() {
        LinearRegression lr = new LinearRegression();
        for (int i = 0; i < 10; i++) {
            lr.al.add(new double[] { i, i * 2 });
        }
        lr.train_test_split();
        assertEquals(7, lr.train_points.size());
        assertEquals(5, lr.test_points.size());
    }

    @Test
    public void testPredict() {
        LinearRegression lr = new LinearRegression();
        lr.slope_ = 2.0;
        lr.intercept_ = 1.0;
        assertEquals(5.0, lr.predict(2.0), 0.001);
    }

    @Test
    public void testCalcMeanSquaredError() {
        LinearRegression lr = new LinearRegression();
        lr.test_points.add(new double[] { 1, 3 });
        lr.test_points.add(new double[] { 2, 5 });
        lr.slope_ = 2.0;
        lr.intercept_ = 1.0;
        lr.calcMeanSquaredError();
        assertEquals(0.0, lr.mean_squared_error_, 0.001);
    }

    @Test
    public void testCalcMeanSquaredError1() {
        LinearRegression lr = new LinearRegression();
        lr.test_points.add(new double[] { 0, 0 });
        lr.test_points.add(new double[] { 0, 0 });
        lr.slope_ = 0.0;
        lr.intercept_ = 0.0;
        lr.calcMeanSquaredError();
        assertEquals(0.0, lr.mean_squared_error_, 0.001);
    }

    @Test
    public void testCalcRSquareScore() {
        LinearRegression lr = new LinearRegression();
        lr.sum_squared_resid = 5.0;
        lr.sum_Y_sq = 20.0;
        lr.calcRSquareScore();
        assertEquals(0.75, lr.r_square_score, 0.001);
    }
}