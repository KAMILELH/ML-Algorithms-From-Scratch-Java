package ml.knn;

import java.util.Arrays;
import ml.core.MLModel;
import ml.metrics.Metrics;

public class KNNClassification extends MLModel{
	private double[][] trainingData;
	private int k;
	
	public KNNClassification(int k) {
		super("KNN Classification");
		this.k=k;
		System.out.println("KNN Classification (k=" + k + ")");
		
	}
	public void train(double[][] dataset) {
		trainingData = dataset;
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
		double somme_target = 0.0;
		double[][] disttarget = new double[n][2]; 
		for(int i=0; i<n; i++) {
			double[] row = trainingData[i];
			double[] rowFeaturs = Arrays.copyOf(row, row.length-1); 
			double dernier_colmn = row[row.length - 1];  
			double distance = euclideanDistance(input, rowFeaturs);  
			disttarget[i][0] = distance;
			disttarget[i][1] = dernier_colmn;
		}
		for (int i = 0; i < n - 1; i++) {
		    for (int j = 0; j < n - i - 1; j++) {
		        if (disttarget[j][0] > disttarget[j+1][0]) {
		            double[] temp = disttarget[j];  
		            disttarget[j] = disttarget[j+1];     
		            disttarget[j+1] = temp;             
		        }
		    }   
		}
		for(int z=0; z<k; z++) {
			somme_target = somme_target + disttarget[z][1];
			
		}
		if(somme_target/k>0.5) {
				return 1.0;
			}
			return 0.0;
	}
	public double score(double[][] testSet) {
		int n = testSet.length;
		double[] yTrue = new double[n];
	    double[] yPred = new double[n];
	    for(int i=0; i<n; i++) {
	    		double[] row = testSet[i];
	    	    double[] features = Arrays.copyOf(row, row.length -1);
	    	    
	    	    yTrue[i] = row[row.length - 1];
	    	    yPred[i] = predict(features);
	    	    if (i < 3) { 
	                System.out.println("Ligne " + i + " -> Vrai: " + yTrue[i] + " | Predit: " + yPred[i]);
	            }
	    }
	    return  Metrics.accuracy(yTrue, yPred);
	}
}
