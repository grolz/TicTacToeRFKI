package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import de.hbo.diagramm.diagramm2D.PunktDoubleDouble;
import de.hbo.tictactoe.TicTacToe;

public class TTTUtil {
    
	public static int[] features (TicTacToe ttt, int next) {
		int[] erg = new int[ttt.getBoard().length*2];
		for (int i = 0; i < ttt.getBoard().length; i++) erg[i] = ttt.getBoard()[i];
		for (int i = 0; i < ttt.getBoard().length; i++) erg[ttt.getBoard().length+i] = i==next?1:0;
		return erg;
    }
    
	public static double[] predict(DQN<?> dqn, TicTacToe ttt) {
    	
    	double[] erg = new double[ttt.getBoard().length];
    	
    	for (int i = 0; i < ttt.getAvailableMovesBinary().length; i++) if (ttt.getAvailableMovesBinary()[i]==1) {
			
    		int[] features = features(ttt, i);

	    	float[] input = new float[features.length];
	    	for (int j = 0; j < features.length; j++) input[j]=features[j];


    		erg[i] = dqn.output(new NDArray(input)).data().asDouble()[0];
    		
    	} else erg[i] = 0;
    	
    	return erg;
		
    }
    
	public static void train(DQN<?> dqn, List<HistoryMap> history) {
    	
    	int batchSize = 500;
    	
    	float[][] featureBatch = new float[batchSize][history.get(0).keySet().toArray(new int [0][0])[0].length];
    	float[] []labelBatch = new float[batchSize][1];
    	
    	for (int j = 0; j < batchSize; j++) {
    		
    		HistoryMap current = history.get(new Random(System.currentTimeMillis()).nextInt(history.size()));
    		int[] key = current.keySet().toArray(new int [0][0])[new Random(System.currentTimeMillis()).nextInt(current.keySet().size())];
    		
    		float[] features = new float[key.length];
    		for (int i = 0; i < key.length; i++) features[i] = key[i];
    		
    		float label = 0f;
    		for (Double each : current.get(key)) label+=each;
    		label/=current.get(key).size();

    		featureBatch[j] = features;
			labelBatch[j] = new float[] {label};
    	}
    	
		dqn.fit(new NDArray(featureBatch), new NDArray(labelBatch));
	        	
    }    
    
	public static double evaluateToHistory(DQN<?> dqn, HistoryMap history) {
    	
    	double mse = 0d;
    	
    	for (int[] key : history.keySet()) {
    		
    		float[] features = new float[key.length];
    		for (int i = 0; i < key.length; i++) features[i] = key[i];
    		
    		float label = 0f;
    		for (Double each : history.get(key)) label+=each;
    		label/=history.get(key).size();
    		
    		mse += Math.pow(dqn.output(new NDArray(features)).data().asDouble()[0]-label, 2d);
    		//mse += Math.pow(historyTable.get2(key)-label, 2d);
    	}
    	
    	return mse/history.size();
	        	
    }    
    
    public static double maxValue(double[] values, int[] availableBinary, int sign) {
    	return values[maxValueIndex(values, availableBinary, sign)];
    }
    
	public static int maxValueIndex(double[] values, int[] availableBinary, int sign) {
    	
    	int erg = -1;
    	double maxValue = Double.NaN;
    	
    	for (int i = 0; i < values.length; i++) if (availableBinary[i]==1 && (erg==-1 || sign*values[i] > maxValue)) {
    		erg = i;
    		maxValue = sign*values[i];
    	}
    	
    	return erg;
    }
    
    // *******************************************************************************************************************************************************************************************
       
	public static String trimDouble(double d, int length) {
    	String erg = Double.toString(d);
    	if (erg.length() > length) erg = erg.substring(0, length);
    	while (erg.length() < length) erg += (erg.indexOf(".") < 0)? "." : "0";
    	return erg;
    }
    
    public static String trimInt(int i, int length) {
    	String erg = Integer.toString(i);
    	while (erg.length() < length) erg = " " + erg;
    	return erg;
    }
	
	public static String toString(int[] i) {
		String erg = "[";
		if (i!=null) for (int j = 0; j < i.length; j++) erg+=(j>0?", ":"") + Integer.toString(i[j]);
		erg+="]";
		return erg;
	}
    
	public static TicTacToe move(TicTacToe ttt, int move) {
		TicTacToe nextTTT = new TicTacToe(ttt.getBoard());
		if (!nextTTT.move(move)) {
			System.out.println(nextTTT.toString());
			throw new RuntimeException("couldn't make next move: " + move);
		}
		return nextTTT;
	}
	
	// *******************************************************************************************************************************************************************************************
    
	public static List<PunktDoubleDouble> average(List<PunktDoubleDouble> original, int length) {
    	List<PunktDoubleDouble> erg = new ArrayList<>();
    	for (int i = 0; i < original.size(); i++) {
    		double sum = 0d;
    		double number = 0d;
    		for (int j = 0; j < length; j++) if (i-j >= 0) {
    			sum+=original.get(i-j).getY();
    			number++;
    		}
    		erg.add(new PunktDoubleDouble(null, null, original.get(i).getX(), sum/number));
    	}
    	return erg;
    }
}
