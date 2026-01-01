package ml.metrics;

public class Metrics {
	public static double r2Score(double[] yTrue, double[] yPred) {
		int n = yTrue.length;
		double u = 0.0;
		double v = 0.0;
		double sum = 0.0;
		for (double val: yTrue ) {
			sum = sum + val; 
		}
		double moyenne = sum / n;
		for (int i=0; i<n; i++) {
			u = u + Math.pow((yTrue[i] - yPred[i]), 2);
			v = v + Math.pow((yTrue[i] - moyenne), 2);
		}
		double R2 = 1 - u/v;
		return R2;
	}
	public static double accuracy(double[] yTrue, double[] yPred) {
		int pred_correct = 0;
		for(int i=0; i<yTrue.length; i++) {
			//On convertit la prédiction (ex: 0.9) en classe entière (ex: 1)
			long classPredit = Math.round(yPred[i]);
			if (yTrue[i] == classPredit) {
				pred_correct ++;
			}
		}
		double ratio = pred_correct/yTrue.length; 
		return ratio;
	}
	 public static double mse(double[] yTrue, double[] yPred) {
		 int n = yTrue.length;
			double sum = 0.0;
			for(int i=0; i<n; i++) {
				sum = sum + Math.pow((yPred[i] - yTrue[i]), 2);
			}
			double MSE = (1/n)*sum;
			return MSE;
	 }
}
