package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import de.hbo.tictactoe.TicTacToe;

public class TTTTrainer4 {

    public static void main(String[] args) throws Exception {
 
        // ------------------------------------------------------------------------------------------------------------------
        // MODEL SETUP
    	
        System.out.print("Setup model:");
      
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(TTTTrainer2.class.getClassLoader().getResourceAsStream("experiment/setting2xxl_100000.zip"));
		
      
        // ******************************************************************************************************************************************************************
        
        model.init();
        DQN<?> dqn = new DQN<>(model);
        System.out.println("done");
        

        // ------------------------------------------------------------------------------------------------------------------
        // TRAINING
        
        System.out.println("Train model...."); 
    
    	int[] results = new int[] {0, 0, 0}; 
    	Evaluator[] evaluators = new Evaluator[] {new PerfectEvaluator(), new DQNEvaluator(dqn)};
    	
    	int index;
    	List<TicTacToe> ttts; 
    	
    	index = -1; // perfect fängt an
    	ttts = new ArrayList<>();
    	ttts.add(new TicTacToe());
    	while (evaluators[++index%2].evaluate(ttts));
		for (TicTacToe ttt : ttts) switch (ttt.isWin()) {
			case  1: results[0] += 1; break;
			case  0: results[2] = results[2]+1; break;
    		case -1: results[1] += 1;
    	}
    	
    	index = 0; // dqn fängt an
    	ttts = new ArrayList<>();
    	ttts.add(new TicTacToe());
    	while (evaluators[++index%2].evaluate(ttts));
		for (TicTacToe ttt : ttts) switch (ttt.isWin()) {
		case  1: results[1] += 1; break;
		case  0: results[2] = results[2]+1; break;
		case -1: results[0] += 1;
	}
    		
    	System.out.println(trimInt(results[0], 3) + " : " + trimInt(results[2], 3) + " : " + trimInt(results[1], 3));
        
        System.out.println("... done.");
    }
    
    private static interface Evaluator {public boolean evaluate(List<TicTacToe> ttts);}
    
    private static class PerfectEvaluator implements Evaluator {
    	
    	private final Map<String, int[]> table = new HashMap<>();

    	public PerfectEvaluator() throws Exception {
	        try (BufferedReader reader = new BufferedReader(new FileReader(new File("src/main/resources/classification/tictactoe.txt")))) {
	        	while (reader.ready()) {
	        		String line = reader.readLine();
	        		if (line == null || line.length() == 0) break;
	        		
	        		String key = line.substring(0, line.indexOf('|'));
	        		String value = line.substring(line.indexOf('|')+1);
	        		List<Integer> values = new ArrayList<>();
	        		for (int i = 0; i < value.length(); i++) if (value.charAt(i)=='1') values.add(i);
	        		int[] v = new int[values.size()];
	        		for (int i = 0; i < values.size(); i++) v[i] = values.get(i).intValue();
	        		
	        		table.put("[" + key.substring(0, key.lastIndexOf(',')) + "]", v);
	        	}
	        }
    	}

		@Override
		public boolean evaluate(List<TicTacToe> ttts) {
    		
			boolean erg = false;
    		
			for (int i = ttts.size()-1; i >= 0; i--) if (!ttts.get(i).isFinal()) {
    			
    			List<TicTacToe> newttts = new ArrayList<>();
    			
    			for (int j : table.get(TTTUtil.toString(ttts.get(i).getBoard()))) {
        			TicTacToe ttt = new TicTacToe(ttts.get(i).getBoard());
    				ttt.move(j);
    				newttts.add(ttt);
    				erg = !ttts.get(i).isFinal() || erg;
    			}
    			
    			ttts.remove(i);
    			ttts.addAll(newttts);
    		}
			
    		return erg;
		}
    	
    }
    
    private static class DQNEvaluator implements Evaluator {
    	
    	private DQN<?> dqn;
    	
    	public DQNEvaluator(DQN dqn) {this.dqn = dqn;}
    	
    	@Override
		public boolean evaluate(List<TicTacToe> ttts) {
    		boolean erg = false;
    		for (int i = ttts.size()-1; i >= 0; i--) if (!ttts.get(i).isFinal()) {
    			ttts.get(i).move(maxValueIndex(predict(this.dqn, ttts.get(i)), ttts.get(i).getAvailableMovesBinary(), ttts.get(i).getNext()));
    			erg = !ttts.get(i).isFinal() || erg;
    		}
    		return erg;
    	}
    }
    
    // *******************************************************************************************************************************************************************************************
    
    private static int[] features (TicTacToe ttt, int next) {
		int[] erg = new int[ttt.getBoard().length*2];
		for (int i = 0; i < ttt.getBoard().length; i++) erg[i] = ttt.getBoard()[i];
		for (int i = 0; i < ttt.getBoard().length; i++) erg[ttt.getBoard().length+i] = i==next?1:0;
		return erg;
    }
    
    //private static HistoryTable historyTable = new HistoryTable();
    
    private static double[] predict(DQN<?> dqn, TicTacToe ttt) {
    	
    	double[] erg = new double[ttt.getBoard().length];
    	
    	for (int i = 0; i < ttt.getAvailableMovesBinary().length; i++) if (ttt.getAvailableMovesBinary()[i]==1) {
			
    		int[] features = features(ttt, i);

	    	float[] input = new float[features.length];
	    	for (int j = 0; j < features.length; j++) input[j]=features[j];


    		erg[i] = dqn.output(new NDArray(input)).data().asDouble()[0];
    		//erg[i] = historyTable.get2(features);
    		
    	} else erg[i] = 0;
    	
    	return erg;
		
    }
    
    // *******************************************************************************************************************************************************************************************
    
    private static double maxValue(double[] values, int[] availableBinary, int sign) {
    	return values[maxValueIndex(values, availableBinary, sign)];
    }
    
    private static int maxValueIndex(double[] values, int[] availableBinary, int sign) {
    	
    	int erg = -1;
    	double maxValue = Double.NaN;
    	
    	for (int i = 0; i < values.length; i++) if (availableBinary[i]==1 && (erg==-1 || sign*values[i] > maxValue)) {
    		erg = i;
    		maxValue = sign*values[i];
    	}
    	
    	return erg;
    }
    
    // *******************************************************************************************************************************************************************************************
       
    private static String trimDouble(double d, int length) {
    	String erg = Double.toString(d);
    	if (erg.length() > length) erg = erg.substring(0, length);
    	while (erg.length() < length) erg += (erg.indexOf(".") < 0)? "." : "0";
    	return erg;
    }
    
    private static String trimInt(int i, int length) {
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
    
	private static TicTacToe move(TicTacToe ttt, int move) {
		TicTacToe nextTTT = new TicTacToe(ttt.getBoard());
		if (!nextTTT.move(move)) {
			System.out.println(nextTTT.toString());
			throw new RuntimeException("couldn't make next move: " + move);
		}
		return nextTTT;
	}
    
}
