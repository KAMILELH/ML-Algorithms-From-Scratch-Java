package ml.app;

import ml.preprocessing.MinMaxScaler;
import ml.preprocessing.StandardScaler;
import ml.model_selection.DataUtils;
import ml.knn.*;
import ml.linear.*;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		double[][] dataset_reg = {
		        {1.0, 10.0},
		        {2.0, 20.0},
		        {3.0, 30.0},
		        {4.0, 40.0},
		        {5.0, 50.0},
		        {6.0, 60.0},
		        {7.0, 70.0},
		        {8.0, 80.0},
		        {9.0, 90.0},
		        {10.0, 100.0}
		    };
		double[][] dataset_class = {
					    
					    {8.0, 8.5, 1.0},
					    {7.5, 7.8, 1.0},
					    {8.2, 8.1, 1.0},
					    {7.8, 8.3, 1.0},
					    {8.5, 7.9, 1.0},
					    {7.2, 8.0, 1.0},
					    {8.1, 8.4, 1.0},
					    {7.9, 7.7, 1.0},
					    {8.3, 8.2, 1.0},
					    {7.6, 8.1, 1.0},
					    
					    {2.0, 2.5, 0.0},
					    {1.5, 1.8, 0.0},
					    {2.2, 2.1, 0.0},
					    {1.8, 2.3, 0.0},
					    {2.5, 1.9, 0.0},
					    {1.2, 2.0, 0.0},
					    {2.1, 2.4, 0.0},
					    {1.9, 1.7, 0.0},
					    {2.3, 2.2, 0.0},
					    {1.6, 2.1, 0.0}
					    
		};
		DataUtils.SplitResult res = DataUtils.trainTestSplit(dataset_reg, 0.2, 42L);
		DataUtils.SplitResult res1 = DataUtils.trainTestSplit(dataset_class, 0.2, 50L);
		
		StandardScaler regScaled = new StandardScaler();
		regScaled.fit(res.trainSet);  //normaliser les features
		res.trainSet = regScaled.transform(res.trainSet);
		res.testSet = regScaled.transform(res.testSet);
		
		
		MinMaxScaler regNormal = new MinMaxScaler();
		regNormal.fit(res1.trainSet);
		res1.trainSet = regNormal.transform(res1.trainSet);
		res1.testSet = regNormal.transform(res1.trainSet);
		
		LinearRegression linearReg = new LinearRegression(0.001, 10000);
		KNNRegression KnnReg = new KNNRegression(3);
		KNNClassification KnnClass = new KNNClassification(3);
		
		//Linear Regression
		System.out.println("◆◆◆◆MODELE DE REGRESSION LINEAR◆◆◆◆");
		linearReg.printStatus();
		linearReg.train(res.trainSet);
		double s1 = linearReg.score(res.testSet);
		System.out.println("Score (R2) de Regression Lineare = "+s1);
		System.out.println("");
		
		//KNN Regression
		System.out.println("◆◆◆◆MODELE DE KNN REGRESSION◆◆◆◆");
		KnnReg.printStatus();
		KnnReg.train(res.trainSet);
		double s2 = KnnReg.score(res.testSet);
		System.out.println("Score (R2) de KNN Regression = "+s2);
		System.out.println("");
		
		//KNN classification
		System.out.println("◆◆◆◆MODELE DE KNN CLASSIFICATION◆◆◆◆");
		KnnClass.printStatus();
		KnnClass.train(res1.trainSet);
		double s3 = KnnClass.score(res1.testSet);
		System.out.println("Score (accuracy) du modèle KNN Classification = "+s3);
		
		
	}

}
