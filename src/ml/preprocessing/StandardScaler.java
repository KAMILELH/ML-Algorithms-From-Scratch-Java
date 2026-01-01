package ml.preprocessing;

public class StandardScaler implements Preprocessor {
	private double[] mean;
	private double[] std;
	//private boolean fitted;
	
	@Override
	public void fit(double[][] dataset) {
		int rows = dataset.length;
		int cols = dataset[0].length;
		this.mean = new double[cols - 1]; //pour que this.mean ne soit pas null  
		this.std = new double[cols - 1];
		
		for (int j=0; j<cols-1; j++) {
			double sum = 0.0;
			for(int i=0; i<rows; i++) {
				sum = sum + dataset[i][j];
			}
			this.mean[j] = sum / rows;
			double sumVariance = 0.0;
			for (int i=0; i<rows; i++) {
				sumVariance = sumVariance + Math.pow(dataset[i][j] - this.mean[j], 2);
			}
			this.std[j] = Math.sqrt(sumVariance / rows);
		}
	}
	@Override
	public double[][] transform(double[][] dataset){
		int rows = dataset.length;
		int cols = dataset[0].length;
		double[][] z = new double[rows][cols];
		for(int i=0; i<rows; i++) {
			for(int j=0; j<cols-1; j++) {
				if(this.std[j]==0) {
					z[i][j] = 0.0;
				}else {
					z[i][j] = (dataset[i][j] - mean[j]) / std[j];
				}
			}
			// copier y sans la modifier 
			z[i][cols-1] = dataset[i][cols-1];
			
		}
		return z;
	}
	@Override
	public double[][] fitTransform(double[][] dataset) {
	    fit(dataset);
	    return transform(dataset);
	}
}
