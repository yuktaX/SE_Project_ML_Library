import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LogisticRegressionTest {

    @Test
    void predictClass() {
        assertEquals(0, LogisticRegression.predictClass(new double[]{0.0, 0.0}));
    }
}