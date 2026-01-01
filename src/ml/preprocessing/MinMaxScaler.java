package ml.preprocessing;

public class MinMaxScaler implements Preprocessor{
	private double[] min;
	private double[] max;
	//private boolean fitted;
	
	@Override
	public void fit(double[][] dataset) {   //à verifier
		for(int i=0; i<dataset.length; i++) {
			for(int j=0; j<dataset[0].length - i - 1; j++) {
				if(dataset[i][j]<dataset[i][j+1]) {
					min = dataset[j];
				}else if(dataset[i][j]>dataset[i][j+1]){
					max = dataset[j];
				}
			}
		}
	}
	@Override
	public double[][] transform(double[][] dataset) {
	    int rows = dataset.length;
	    int cols = dataset[0].length;
	    double[][] normalized = new double[rows][cols];

	    for (int i = 0; i < rows; i++) {
	        for (int j = 0; j < cols; j++) {
	            if (max[j] == min[j] ) {
	                normalized[i][j] = 0.0;
	            } else {
	                normalized[i][j] = (dataset[i][j] - min[j]) / (max[j] - min[j]);
	            }
	        }
	    }
	    return normalized;
	}
	@Override
	public double[][] fitTransform(double[][] dataset) {
	    fit(dataset);
	    return transform(dataset);
	}

}
