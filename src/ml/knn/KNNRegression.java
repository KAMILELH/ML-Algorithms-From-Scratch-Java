package ml.knn;

import ml.core.MLModel;
import ml.metrics.Metrics;


public class KNNRegression extends MLModel {
	private double[][] trainingData;
	private int k; 
	
	public KNNRegression(int k) {
		super("KNN Regression");
		if (k>0) {
			System.out.println("k est positive");
		}else {
			k =- k;
			System.out.println("k est change en int positive");
		}
		
		System.out.println("KNN Regression (k=" + k + ")");
		this.k=k;
	}
	public void train(double[][] dataset) {
		trainingData=dataset;

	}
	private double euclideanDistance(double[] a, double[] b) {
		double sum = 0.0;
		for (int j = 0; j < a.length; j++) {
			double diff = a[j]- b[j];
			sum += diff * diff;
		}
		return Math.sqrt(sum);
	}
	
	public double predict(double[] input) {
		int n = trainingData.length;
		int cols = trainingData[0].length;
		double somme = 0.0;
		double[][] distance = new double[n][2]; //2 = dist et y
		for(int i=0; i<n; i++) {
			//Extraire la ligne complete
			double[] row = trainingData[i];
			//extraire la derniere colonne
			double target = row[cols - 1]; 
			//extraire les features
			double[] features = new double[cols - 1];
			for(int j=0; j<cols-1; j++) {
				features[j] = row[j];
			}
			double dist = euclideanDistance(input, features);
			distance[i][0] = dist;
			distance[i][1] = target;
			
		}
		for (int i = 0; i < n - 1; i++) {
		    for (int j = 0; j < n - i - 1; j++) {
		        if (distance[j][0] > distance[j+1][0]) {
		            double[] temp = distance[j];  
		            distance[j] = distance[j+1];     
		            distance[j+1] = temp;             
		        }
		    }   
		}
		for (int z=0; z<this.k; z++) {
		    	    somme += distance[z][1];
		    }
		return somme/this.k;
    }
	
	public double score(double[][] testSet) {
		int n = testSet.length;
		int cols = testSet[0].length;
		double[] yTrue = new double[n];
	    double[] yPred = new double[n];
	    for(int i=0; i<n; i++) {
	    	
	    	yTrue[i] = testSet[i][cols-1];
	    	//toute les cols sauf la derniere
	    	double[] input = new double[cols-1];
	    	for(int j=0; j<cols-1; j++) {
	    		 input[j] = testSet[i][j];
	    	}
	    	//predire ligne par ligne
	    	yPred[i] = predict(input);
	    	
	    }
	    return  Metrics.r2Score(yTrue, yPred);
	}
}

