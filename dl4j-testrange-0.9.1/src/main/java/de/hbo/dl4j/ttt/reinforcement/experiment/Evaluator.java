package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.rl4j.network.dqn.DQN;

import de.hbo.tictactoe.TicTacToe;

public interface Evaluator {
	
	public boolean evaluate(List<TicTacToe> ttts);

	public static class PerfectEvaluator implements Evaluator {
		
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
	
	public static class DQNEvaluator implements Evaluator {
		
		private DQN<?> dqn;
		
		public DQNEvaluator(DQN<?> dqn) {this.dqn = dqn;}
		
		@Override
		public boolean evaluate(List<TicTacToe> ttts) {
			boolean erg = false;
			for (int i = ttts.size()-1; i >= 0; i--) if (!ttts.get(i).isFinal()) {
				ttts.get(i).move(TTTUtil.maxValueIndex(TTTUtil.predict(this.dqn, ttts.get(i)), ttts.get(i).getAvailableMovesBinary(), ttts.get(i).getNext()));
				erg = !ttts.get(i).isFinal() || erg;
			}
			return erg;
		}
	}
}