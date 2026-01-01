package ml.model_selection;
import java.util.Random;
public class DataUtils {
	public static class SplitResult {
		public double[][] trainSet;
		public double[][] testSet;
	}	
	public static SplitResult trainTestSplit(double[][] dataset,
				double testRatio, 
				long seed) {
				int rows = dataset.length;
				int cols = dataset[0].length;
				if(dataset.length == 0) {
					System.out.println("dataset est vide");
					System.exit(-1);
				}
					int[] indices = new int[rows];
					for(int i=0; i<rows; i++) {
						indices[i] = i;
					}
					Random rand = new Random(seed);

					
					for (int i = rows - 1; i > 0; i--) {
					    // On tire un index aléatoire j entre 0 et i
					    int j = rand.nextInt(i + 1);
					    // On échange indices[i] avec indices[j]
					    int temp = indices[i];
					    indices[i] = indices[j];
					    indices[j] = temp;
					}					
					int testSize = (int) (testRatio * rows);
					int trainSize = rows - testSize;
					
					SplitResult split = new SplitResult();
					// il faut declarer la taille des deux tableaux
					split.trainSet = new double[trainSize][cols];
					split.testSet = new double[testSize][cols];
					
					for(int i=0; i<trainSize; i++) {
						int indiceReel = indices[i];
						for(int j=0; j<cols; j++) {
							split.trainSet[i][j] = dataset[indiceReel][j];
						}
					}
					for (int i=0; i<testSize; i++) {
						int indiceReel = indices[trainSize + i];
						for(int j=0; j<cols; j++) {
							split.testSet[i][j] = dataset[indiceReel][j];
						}
					}
					
					return split;
				
		}
	
	 
}
