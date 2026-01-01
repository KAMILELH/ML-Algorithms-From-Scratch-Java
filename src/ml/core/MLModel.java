package ml.core;

public abstract class MLModel {
	protected String name;  //nom de modele
	
	public MLModel(String name) {
		this.name = name;
	}
	
	public void printStatus() {
		System.out.println("Modele: "+name+"(pret).");
	}
	
	public abstract void train(double[][] dataset);
	
	public abstract double predict(double[] input);
	
	public double[] predict(double[][] inputs) {  
		double[] predictions = new double[inputs.length];
		for(int i=0; i<inputs.length; i++) {
			predictions[i]=this.predict(inputs[i]);
			}
		return predictions;
		}
	public abstract double score(double[][] testSet);
	
	public String getName() {
		return this.name;
	}

}
