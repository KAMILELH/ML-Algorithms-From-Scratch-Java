package ml.linear;

import ml.core.MLModel;
import ml.metrics.Metrics;

public class LinearRegression extends MLModel {
	private double slope;
	private double intercept;
	private double learningRate;
	private int numEpochs;
	
	
	
	public LinearRegression() { 
		super("Linear Regression");
		slope = 0.0;   //m
		intercept = 0.0;  //b
		learningRate = 0.01;  //alpha
		numEpochs = 1000;    //n_iter
	}
	
	public LinearRegression(double learningRate, int numEpochs) {
		super("Linear Regression");
		this.learningRate = learningRate;
		this.numEpochs = numEpochs;
	}
	
	private void initializeParameters() {
		slope = 0.0;
		intercept = 0.0;
	}
	private boolean isDatasetValid(double[][] dataset) {
		if(dataset != null && dataset.length > 0) {
			return true;
		}
		else {
			return false;
		}
	}
		private double[] computeGradients(double[][] dataset) {
			double sum_m = 0.0;
			double sum_b = 0.0;
			int n = dataset.length;
			for(int i=0; i<dataset.length; i++) {
				double x = dataset[i][0];
				double y = dataset[i][1];
				
				double model = this.slope * x + this.intercept;
				double A = (model - y);
				 sum_m = sum_m + (A * x);
				 sum_b = sum_b + A;
				
			}
			double gradSlope = (2.0/n)*sum_m; 
			double gradIntercept = (2.0/n)*sum_b;
			
			return  new double[] {gradSlope, gradIntercept}; 
		}
		private void updateParameters(double gradSlope, double gradIntercept) {
			this.slope = this.slope - learningRate*gradSlope;
			this.intercept = this.intercept - learningRate*gradIntercept;
				        
		}   
		private double computeCost(double[][] dataset) {
			int n = dataset.length;
			double B = 0.0;
			for(int i=0; i<dataset.length; i++) {
				double x = dataset[i][0];
				double y = dataset[i][1];
				double model = this.slope*x + this.intercept;
				B = B + Math.pow((model - y),2);
			}
			double MSE = (1.0/n)*B;     //MSE = (1/n)*B  division de 2 entiers donne un entier donc il donne tjrs 0.0
			return MSE;               //alors le modele n'apprend rien
		}
		private void gradientDescentLoop(double[][] dataset) {
			for(int epoch=0; epoch<numEpochs; epoch++) {
				double[] grad = computeGradients(dataset);
				double gradSlope = grad[0];
				double gradIntercept = grad[1];
				updateParameters(gradSlope, gradIntercept);
				computeCost(dataset);
				if (epoch % 1000 ==0) {
					System.out.println("Le cout MSE n°"+epoch+": "+computeCost(dataset));
				}				
			}
		}
		public void train(double[][] dataset) {
			isDatasetValid(dataset);
			initializeParameters();
			gradientDescentLoop(dataset);
		}
		
		public double predict(double[] input) {
			double x = input[0];
			return this.slope*x + intercept;
		}
	    
		public double score(double[][] testSet) {
			int n = testSet.length;
			double[] yTrue = new double[n];
		    double[] yPred = new double[n];

		    for (int i=0; i<n; i++) {
		        
		        yTrue[i] = testSet[i][testSet[i].length - 1];
		        double[] input = { testSet[i][0] }; //Enveloppe les feature dans un nouveau tableau de type double[]
		        yPred[i] = predict(input);          
		    }
		    return Metrics.r2Score(yTrue, yPred);
			
		}
	}


